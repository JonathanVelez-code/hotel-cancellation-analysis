-- Seasonality: bookings, cancellations and average daily rate by arrival month and hotel,
-- with each month's cancellation rate compared to the same month a year earlier.
WITH m AS (
  SELECT hotel,
         DATE_TRUNC('month', arrival_date)                     AS arrival_month,
         COUNT(*)                                              AS bookings,
         ROUND(100.0 * AVG(is_canceled), 1)                    AS cancel_rate_pct,
         ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2) AS avg_daily_rate,
         ROUND(SUM(CASE WHEN is_canceled = 0 THEN room_revenue END), 0) AS realized_revenue
  FROM bookings
  GROUP BY hotel, DATE_TRUNC('month', arrival_date)
)
SELECT *,
       LAG(cancel_rate_pct, 12) OVER (PARTITION BY hotel ORDER BY arrival_month) AS cancel_rate_same_month_last_year
FROM m
ORDER BY hotel, arrival_month;
