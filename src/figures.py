"""Charts for the README and for presenting the findings."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.model import CATEGORICAL  # noqa: E402

BLUE, ORANGE, GRAY = "#1f4e79", "#c55a11", "#a6a6a6"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def cancel_by_lead_time():
    d = pd.read_csv(config.POWERBI / "sql_02_cancellations_by_lead_time.csv")
    order = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (hotel, color) in enumerate((("City Hotel", BLUE), ("Resort Hotel", ORANGE))):
        h = d[d.hotel == hotel].set_index("lead_time_bucket").reindex(order)
        x = [j + (i - 0.5) * 0.38 for j in range(len(order))]
        ax.bar(x, h["cancel_rate_pct"], 0.38, color=color, label=hotel)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_xlabel("How far ahead the booking was made")
    ax.set_title("Bookings made further ahead cancel far more often")
    ax.legend(frameon=False)
    save(fig, "cancel_by_lead_time.png")


def cancel_by_segment():
    d = pd.read_csv(config.POWERBI / "sql_03_cancellations_by_segment.csv")
    d = d[d.dimension.isin(["market_segment", "deposit_type"])].sort_values("cancel_rate_pct")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [ORANGE if dim == "deposit_type" else BLUE for dim in d.dimension]
    ax.barh(d["value"], d["cancel_rate_pct"], color=colors)
    ax.set_xlabel("Cancellation rate (%)")
    ax.set_title("Cancellation rate by market segment (blue) and deposit type (orange)")
    save(fig, "cancel_by_segment.png")


def roc_curves(t):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    res = json.loads((config.OUTPUTS / "model_results.json").read_text())["models"]
    for col, name, color in (("p_xgb", "XGBoost", BLUE), ("p_logreg", "Logistic regression", ORANGE),
                             ("p_baseline", "Baseline (hotel x lead time rate)", GRAY)):
        key = name
        fpr, tpr, _ = roc_curve(t["is_canceled"], t[col])
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {res[key]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Model comparison, Jan-Aug 2017 test set")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "roc_curves.png")


def gains():
    g = pd.read_csv(config.OUTPUTS / "gains_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, color in (("XGBoost", BLUE), ("Baseline", GRAY)):
        s = g[g.model == name]
        ax.plot(s["pct_bookings_targeted"], s["pct_cancellations_captured"], color=color, lw=2, label=name)
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls="--", label="Random")
    ax.axvline(20, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("% of bookings targeted (riskiest first)")
    ax.set_ylabel("% of cancellations captured")
    ax.set_title("Targeting the riskiest 20% catches 44% of cancellations")
    ax.legend(frameon=False)
    save(fig, "gains_curve.png")


def weekly_overbooking():
    w = pd.read_csv(config.OUTPUTS / "weekly_overbooking.csv", parse_dates=["week"])
    w = w[w.hotel == "City Hotel"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(w["week"], w["actual_cancellations"], color="black", lw=1.5, label="Actual cancellations")
    ax.plot(w["week"], w["expected_cancellations"], color=BLUE, lw=1.5, label="Model forecast")
    ax.fill_between(w["week"], 0, w["safe_overbook_rooms"], color=ORANGE, alpha=0.25,
                    label="Safe overbooking (95% confidence)")
    ax.set_ylabel("Rooms per arrival week")
    ax.set_title("City Hotel: weekly cancellations, forecast and safe overbooking level")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "weekly_overbooking_city.png")


def feature_importance():
    f = pd.read_csv(config.OUTPUTS / "feature_importance.csv")

    def original(name):
        for c in CATEGORICAL:
            if name.startswith(c + "_"):
                return c
        return name
    agg = f.assign(feature=f["feature"].map(original)).groupby("feature")["importance"].sum().sort_values().tail(12)
    labels = {"agent_top": "travel agent", "arrival_month_name": "arrival month"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh([labels.get(i, i.replace("_", " ")) for i in agg.index], agg.values, color=BLUE)
    ax.set_title("What drives cancellation risk")
    ax.set_xlabel("XGBoost importance")
    save(fig, "feature_importance.png")


def run(test: pd.DataFrame):
    cancel_by_lead_time()
    cancel_by_segment()
    roc_curves(test)
    gains()
    weekly_overbooking()
    feature_importance()
