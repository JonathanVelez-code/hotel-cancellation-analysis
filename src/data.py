"""Download and clean the Hotel Booking Demand data (119,390 real bookings, 2015-2017).

Cleaning rules (each one is counted in outputs/data_quality_report.md):
- "NULL" text -> missing; children missing -> 0; country missing -> "Unknown"; agent/company missing -> "none"
- Meal "Undefined" -> "SC" (the dataset paper says they mean the same: no meal package)
- Drop bookings with zero guests (adults + children + babies = 0)
- Drop impossible prices (negative ADR) and the single extreme outlier (ADR > 1,000)
- Identical rows are KEPT: there's no booking ID, so identical rows can be separate rooms in
  the same group booking. They're counted in the report instead.
"""
import json
import urllib.request

import pandas as pd

from src import config

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def download() -> None:
    if not config.RAW_FILE.exists():
        urllib.request.urlretrieve(config.DATA_URL, config.RAW_FILE)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    report = {"raw_rows": len(d), "identical_rows_kept": int(d.duplicated().sum())}

    report["children_missing_filled"] = int(d["children"].isna().sum())
    d["children"] = d["children"].fillna(0).astype(int)
    report["country_missing_filled"] = int(d["country"].isna().sum())
    d["country"] = d["country"].fillna("Unknown")
    for col in ("agent", "company"):
        d[col] = d[col].fillna("none").astype(str).str.replace(r"\.0$", "", regex=True)

    report["meal_undefined_to_SC"] = int((d["meal"] == "Undefined").sum())
    d["meal"] = d["meal"].replace("Undefined", "SC")

    no_guests = (d["adults"] + d["children"] + d["babies"]) == 0
    report["zero_guest_rows_dropped"] = int(no_guests.sum())
    d = d[~no_guests]

    bad_adr = (d["adr"] < 0) | (d["adr"] > 1000)
    report["impossible_or_extreme_adr_dropped"] = int(bad_adr.sum())
    d = d[~bad_adr].copy()

    # Derived columns
    d["arrival_date"] = pd.to_datetime(dict(year=d["arrival_date_year"],
                                            month=d["arrival_date_month"].map(MONTHS),
                                            day=d["arrival_date_day_of_month"]))
    d["arrival_month"] = d["arrival_date"].dt.month
    d["booking_date"] = d["arrival_date"] - pd.to_timedelta(d["lead_time"], unit="D")
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["guests"] = d["adults"] + d["children"] + d["babies"]
    d["room_revenue"] = (d["adr"] * d["total_nights"]).round(2)       # revenue if the stay happens
    d["lead_time_bucket"] = pd.cut(d["lead_time"], [-1, 7, 30, 90, 180, 365, 10_000],
                                   labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days",
                                           "181-365 days", "365+ days"]).astype(str)
    d = d.reset_index(drop=True)
    d.insert(0, "booking_id", d.index + 1)

    report["clean_rows"] = len(d)
    report["rows_removed"] = report["raw_rows"] - report["clean_rows"]
    return d, report


def write_report(report: dict) -> None:
    (config.OUTPUTS / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Data quality report", "", "| Check | Rows |", "|---|---:|"]
    lines += [f"| {k.replace('_', ' ')} | {v:,} |" for k, v in report.items()]
    (config.OUTPUTS / "data_quality_report.md").write_text("\n".join(lines) + "\n")


def run() -> tuple[pd.DataFrame, dict]:
    download()
    raw = pd.read_csv(config.RAW_FILE, na_values=["NULL"])
    d, report = clean(raw)
    d.to_parquet(config.DATA_CLEAN / "bookings.parquet", index=False)
    write_report(report)
    return d, report


if __name__ == "__main__":
    _, r = run()
    print(json.dumps(r, indent=2))
