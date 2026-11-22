-- The further ahead a booking is made, the more likely it is to cancel.
SELECT hotel,
       lead_time_bucket,
       COUNT(*)                                                       AS bookings,
       ROUND(100.0 * AVG(is_canceled), 1)                             AS cancel_rate_pct,
       ROUND(SUM(CASE WHEN is_canceled = 1 THEN room_revenue END), 0) AS revenue_lost,
       ROUND(AVG(adr), 2)                                             AS avg_daily_rate
FROM bookings
GROUP BY hotel, lead_time_bucket
ORDER BY hotel, MIN(lead_time);
