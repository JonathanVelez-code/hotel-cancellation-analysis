"""Write Power BI-ready tables to outputs/powerbi/ (see powerbi/BUILD_GUIDE.md)."""
import shutil

import pandas as pd

from src import config


def run(bookings: pd.DataFrame, test: pd.DataFrame) -> None:
    out = config.POWERBI
    cols = ["booking_id", "hotel", "is_canceled", "lead_time", "lead_time_bucket", "booking_date", "arrival_date",
            "arrival_month", "total_nights", "guests", "adr", "room_revenue", "meal", "country", "market_segment",
            "distribution_channel", "is_repeated_guest", "deposit_type", "customer_type", "reserved_room_type"]
    fact = bookings[cols].merge(test[["booking_id", "p_xgb"]].rename(columns={"p_xgb": "cancel_risk"}),
                                on="booking_id", how="left")
    fact["risk_band"] = pd.cut(fact["cancel_risk"], [-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]).astype(str)
    fact.loc[fact["cancel_risk"].isna(), "risk_band"] = "Not scored (training period)"
    fact.to_csv(out / "fact_bookings.csv", index=False, date_format="%Y-%m-%d")

    dates = pd.DataFrame({"date": pd.date_range(fact["booking_date"].min(), fact["arrival_date"].max())})
    dates["year"] = dates["date"].dt.year
    dates["month"] = dates["date"].dt.month
    dates["month_name"] = dates["date"].dt.strftime("%b")
    dates["year_month"] = dates["date"].dt.strftime("%Y-%m")
    dates.to_csv(out / "dim_date.csv", index=False, date_format="%Y-%m-%d")

    for f in ("weekly_overbooking.csv", "gains_curve.csv", "feature_importance.csv"):
        shutil.copy(config.OUTPUTS / f, out / f)
