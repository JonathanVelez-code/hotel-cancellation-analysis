-- Bookings, cancellation rate, realized revenue and revenue lost to cancellations, by hotel.
-- "Lost" = room revenue the cancelled bookings would have brought in (some rooms may have been resold).
SELECT hotel,
       COUNT(*)                                                         AS bookings,
       SUM(is_canceled)                                                 AS cancellations,
       ROUND(100.0 * AVG(is_canceled), 1)                               AS cancel_rate_pct,
       ROUND(SUM(CASE WHEN is_canceled = 0 THEN room_revenue END), 0)   AS realized_revenue,
       ROUND(SUM(CASE WHEN is_canceled = 1 THEN room_revenue END), 0)   AS revenue_lost_to_cancellations,
       ROUND(100.0 * SUM(CASE WHEN is_canceled = 1 THEN room_revenue END) / SUM(room_revenue), 1) AS pct_of_booked_revenue_lost,
       ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)            AS avg_daily_rate
FROM bookings
GROUP BY hotel
ORDER BY hotel;
