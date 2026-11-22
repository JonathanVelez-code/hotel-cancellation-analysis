"""Predict which bookings will cancel, using only what's known when the booking is made.

Leakage checks (columns deliberately EXCLUDED):
- reservation_status, reservation_status_date: they ARE the outcome.
- assigned_room_type: assigned at check-in, so cancelled bookings rarely have a different one.
- booking_changes: keeps counting until check-in or cancellation.
- required_car_parking_spaces: in this data it's essentially only filled in for guests who arrive.
- country: per the dataset's paper, country is often only confirmed at check-in, so cancelled
  bookings default to Portugal (57% "cancellation rate" for PRT vs 24% for everyone else).

Validation is by TIME: train on arrivals Jul 2015-Dec 2016, test on Jan-Aug 2017.
Models: logistic regression (simple, explainable baseline) vs XGBoost.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config

CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
               "deposit_type", "customer_type", "agent_top", "arrival_month_name"]
NUMERIC = ["lead_time", "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights",
           "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
           "previous_bookings_not_canceled", "days_in_waiting_list", "adr", "total_of_special_requests"]
EXCLUDED = ["reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes",
            "required_car_parking_spaces", "country"]


def prepare(d: pd.DataFrame, top_agents=None) -> tuple[pd.DataFrame, list]:
    x = d.copy()
    if top_agents is None:  # learn the 30 most common agents from training data only
        top_agents = x["agent"].value_counts().head(30).index.tolist()
    x["agent_top"] = np.where(x["agent"].isin(top_agents), x["agent"], "other")
    x["arrival_month_name"] = x["arrival_date"].dt.strftime("%b")
    return x[CATEGORICAL + NUMERIC], top_agents


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
                              ("num", num, NUMERIC)])


def models() -> dict:
    return {
        "Logistic regression": Pipeline([("prep", preprocessor(scale=True)),
                                         ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
        "XGBoost": Pipeline([("prep", preprocessor(scale=False)),
                             ("clf", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   eval_metric="logloss", random_state=config.SEED))]),
    }


def metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "precision_at_0.5": round(precision_score(y, pred), 4),
            "recall_at_0.5": round(recall_score(y, pred), 4),
            "brier_score": round(brier_score_loss(y, p), 4)}


def run(d: pd.DataFrame) -> dict:
    train = d[d["arrival_date"] < config.TEST_START]
    test = d[d["arrival_date"] >= config.TEST_START].copy()
    x_train, top_agents = prepare(train)
    x_test, _ = prepare(test, top_agents)

    results, fitted = {}, {}
    for name, m in models().items():
        m.fit(x_train, train["is_canceled"])
        p = m.predict_proba(x_test)[:, 1]
        results[name] = metrics(test["is_canceled"], p)
        test[f"p_{'xgb' if name == 'XGBoost' else 'logreg'}"] = p
        fitted[name] = m

    # Naive baseline: each booking gets its hotel x lead-time-bucket cancellation rate from training
    rate = train.groupby(["hotel", "lead_time_bucket"])["is_canceled"].mean()
    p_base = test.set_index(["hotel", "lead_time_bucket"]).index.map(rate).to_numpy(dtype=float)
    results["Baseline (hotel x lead time rate)"] = metrics(test["is_canceled"], p_base)
    test["p_baseline"] = p_base

    # Feature importance (XGBoost gain, mapped back to readable names)
    xgb = fitted["XGBoost"]
    names = xgb.named_steps["prep"].get_feature_names_out()
    imp = (pd.DataFrame({"feature": names, "importance": xgb.named_steps["clf"].feature_importances_})
           .assign(feature=lambda f: f["feature"].str.replace(r"^(cat|num)__", "", regex=True))
           .sort_values("importance", ascending=False))

    summary = {
        "train": f"arrivals {train['arrival_date'].min().date()} to {train['arrival_date'].max().date()} ({len(train):,} bookings)",
        "test": f"arrivals {test['arrival_date'].min().date()} to {test['arrival_date'].max().date()} ({len(test):,} bookings)",
        "test_cancel_rate": round(float(test["is_canceled"].mean()), 4),
        "excluded_for_leakage": EXCLUDED,
        "models": results,
    }
    (config.OUTPUTS / "model_results.json").write_text(json.dumps(summary, indent=2))
    imp.to_csv(config.OUTPUTS / "feature_importance.csv", index=False)
    test.to_parquet(config.DATA_CLEAN / "test_predictions.parquet", index=False)
    return {"summary": summary, "test": test, "importance": imp, "train": train}


if __name__ == "__main__":
    d = pd.read_parquet(config.DATA_CLEAN / "bookings.parquet")
    out = run(d)
    print(json.dumps(out["summary"], indent=2))
    print(out["importance"].head(12).to_string())
