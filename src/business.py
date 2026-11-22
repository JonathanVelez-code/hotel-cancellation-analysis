"""Turn cancellation predictions into revenue decisions (test period: Jan-Aug 2017 arrivals).

1. Risk targeting: how much of the cancellation problem sits in the riskiest bookings?
2. Overbooking: forecast cancellations per hotel per arrival week, and a "safe" overbooking
   level with only a 5% chance of more guests showing up than expected.
3. Deposit policy: which no-deposit bookings look like the best candidates for a deposit.
"""
import json

import numpy as np
import pandas as pd

from src import config

Z_95 = 1.645  # one-sided 95%


def risk_targeting(t: pd.DataFrame, col: str) -> dict:
    s = t.sort_values(col, ascending=False)
    n = len(s)
    out = {}
    for share in (0.10, 0.20, 0.30):
        top = s.head(int(n * share))
        out[f"top_{int(share * 100)}pct"] = {
            "bookings": len(top),
            "cancel_rate_pct": round(top["is_canceled"].mean() * 100, 1),
            "share_of_all_cancellations_pct": round(top["is_canceled"].sum() / s["is_canceled"].sum() * 100, 1),
            "share_of_lost_revenue_pct": round(
                top.loc[top["is_canceled"] == 1, "room_revenue"].sum()
                / s.loc[s["is_canceled"] == 1, "room_revenue"].sum() * 100, 1),
        }
    return out


def gains_curve(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = t["is_canceled"].sum()
    for col, label in (("p_xgb", "XGBoost"), ("p_logreg", "Logistic regression"), ("p_baseline", "Baseline")):
        s = t.sort_values(col, ascending=False)["is_canceled"].to_numpy()
        cum = np.cumsum(s) / total
        for pct in range(0, 101, 5):
            k = int(len(s) * pct / 100)
            rows.append({"model": label, "pct_bookings_targeted": pct,
                         "pct_cancellations_captured": round(cum[k - 1] * 100 if k else 0, 1)})
    return pd.DataFrame(rows)


def weekly_overbooking(t: pd.DataFrame, train: pd.DataFrame, col: str) -> tuple[pd.DataFrame, dict]:
    t = t.assign(week=t["arrival_date"].dt.to_period("W-SUN").dt.start_time)
    avg_stay_rev = train[train["is_canceled"] == 0].groupby("hotel")["room_revenue"].mean()
    hist_rate = train.groupby("hotel")["is_canceled"].mean()

    w = (t.groupby(["hotel", "week"])
         .agg(bookings=("is_canceled", "size"), actual_cancellations=("is_canceled", "sum"),
              expected_cancellations=(col, "sum"),
              variance=(col, lambda p: float((p * (1 - p)).sum())))
         .reset_index())
    w["baseline_expected"] = w["bookings"] * w["hotel"].map(hist_rate)
    # Safe overbooking: cancellations we're 95% sure will happen (normal approx. to the sum of Bernoullis)
    w["safe_overbook_rooms"] = np.floor(w["expected_cancellations"] - Z_95 * np.sqrt(w["variance"])).clip(lower=0)
    w["overbooked_too_much"] = w["actual_cancellations"] < w["safe_overbook_rooms"]
    w["revenue_recoverable"] = w["safe_overbook_rooms"] * w["hotel"].map(avg_stay_rev)
    w = w[w["bookings"] >= 50]  # skip partial weeks at the edges

    def wape(pred):
        return float((w[pred] - w["actual_cancellations"]).abs().sum() / w["actual_cancellations"].sum() * 100)

    summary = {
        "weeks": int(len(w)),
        "weekly_cancellation_forecast_wape_pct_model": round(wape("expected_cancellations"), 1),
        "weekly_cancellation_forecast_wape_pct_baseline": round(wape("baseline_expected"), 1),
        "safe_overbook_rooms_total": int(w["safe_overbook_rooms"].sum()),
        "weeks_where_overbooking_would_have_been_too_much_pct": round(w["overbooked_too_much"].mean() * 100, 1),
        "revenue_recoverable_if_overbooked_rooms_are_resold": round(float(w["revenue_recoverable"].sum()), 0),
        "note": "Recoverable revenue assumes each overbooked room is resold at that hotel's average stay "
                "revenue. It is an upper bound that depends on enough demand to fill the rooms.",
    }
    return w, summary


def deposit_candidates(t: pd.DataFrame, col: str) -> dict:
    nd = t[t["deposit_type"] == "No Deposit"]
    cutoff = nd[col].quantile(0.80)
    high = nd[nd[col] >= cutoff]
    return {
        "segment": "No-deposit bookings in the top 20% of predicted risk",
        "bookings": len(high),
        "cancel_rate_pct": round(high["is_canceled"].mean() * 100, 1),
        "no_deposit_overall_cancel_rate_pct": round(nd["is_canceled"].mean() * 100, 1),
        "revenue_lost_in_segment": round(float(high.loc[high["is_canceled"] == 1, "room_revenue"].sum()), 0),
        "median_lead_time_days": int(high["lead_time"].median()),
        "most_common_segment": high["market_segment"].mode().iat[0],
    }


def run(test: pd.DataFrame, train: pd.DataFrame) -> dict:
    # XGBoost ranks risk best (higher AUC); for COUNTING cancellations use the better-calibrated
    # model (lower Brier score), because overbooking depends on the probabilities being right.
    res = json.loads((config.OUTPUTS / "model_results.json").read_text())["models"]
    count_col = "p_logreg" if res["Logistic regression"]["brier_score"] <= res["XGBoost"]["brier_score"] else "p_xgb"

    weekly, ob = weekly_overbooking(test, train, count_col)
    summary = {
        "risk_targeting_xgboost": risk_targeting(test, "p_xgb"),
        "risk_targeting_baseline": risk_targeting(test, "p_baseline"),
        "overbooking_model_used": "Logistic regression" if count_col == "p_logreg" else "XGBoost",
        "overbooking": ob,
        "deposit_policy": deposit_candidates(test, "p_xgb"),
    }
    (config.OUTPUTS / "business_results.json").write_text(json.dumps(summary, indent=2, default=str))
    weekly.to_csv(config.OUTPUTS / "weekly_overbooking.csv", index=False, date_format="%Y-%m-%d")
    gains_curve(test).to_csv(config.OUTPUTS / "gains_curve.csv", index=False)
    return summary


if __name__ == "__main__":
    test = pd.read_parquet(config.DATA_CLEAN / "test_predictions.parquet")
    d = pd.read_parquet(config.DATA_CLEAN / "bookings.parquet")
    train = d[d["arrival_date"] < config.TEST_START]
    print(json.dumps(run(test, train), indent=2, default=str))
