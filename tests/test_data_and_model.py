import numpy as np
import pandas as pd

from src.business import risk_targeting
from src.data import clean
from src.model import EXCLUDED, prepare


def raw_rows():
    base = dict(hotel="City Hotel", is_canceled=0, lead_time=10, arrival_date_year=2016,
                arrival_date_month="March", arrival_date_week_number=10, arrival_date_day_of_month=5,
                stays_in_weekend_nights=1, stays_in_week_nights=2, adults=2, children=0.0, babies=0,
                meal="BB", country="PRT", market_segment="Direct", distribution_channel="Direct",
                is_repeated_guest=0, previous_cancellations=0, previous_bookings_not_canceled=0,
                reserved_room_type="A", assigned_room_type="A", booking_changes=0, deposit_type="No Deposit",
                agent=9.0, company=np.nan, days_in_waiting_list=0, customer_type="Transient", adr=100.0,
                required_car_parking_spaces=0, total_of_special_requests=0, reservation_status="Check-Out",
                reservation_status_date="2016-03-08")
    rows = [dict(base) for _ in range(5)]
    rows[1].update(adults=0)                    # zero guests -> dropped
    rows[2].update(adr=-5.0)                    # negative price -> dropped
    rows[3].update(meal="Undefined", children=np.nan, country=np.nan)
    return pd.DataFrame(rows)


def test_clean_rules():
    d, report = clean(raw_rows())
    assert report["zero_guest_rows_dropped"] == 1
    assert report["impossible_or_extreme_adr_dropped"] == 1
    assert len(d) == 3
    assert "Undefined" not in set(d["meal"])
    assert d["children"].isna().sum() == 0 and d["country"].isna().sum() == 0


def test_derived_revenue_and_dates():
    d, _ = clean(raw_rows())
    row = d.iloc[0]
    assert row["total_nights"] == 3
    assert row["room_revenue"] == 300.0
    assert row["booking_date"] == pd.Timestamp("2016-02-24")   # 10 days before 2016-03-05


def test_no_leaky_columns_in_model_features():
    d, _ = clean(raw_rows())
    x, _ = prepare(d)
    assert not set(EXCLUDED) & set(x.columns)


def test_risk_targeting_shares():
    t = pd.DataFrame({"p": [0.9, 0.8, 0.2, 0.1] * 5, "is_canceled": [1, 1, 0, 0] * 5,
                      "room_revenue": [100.0] * 20})
    r = risk_targeting(t, "p")
    assert r["top_10pct"]["cancel_rate_pct"] == 100.0
    assert r["top_30pct"]["share_of_all_cancellations_pct"] == 60.0
