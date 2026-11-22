-- Cancellation rate and lost revenue by market segment, deposit type and repeat-guest status.
-- Uses GROUPING SETS to produce three breakdowns in one pass.
SELECT CASE WHEN GROUPING(market_segment) = 0 THEN 'market_segment'
            WHEN GROUPING(deposit_type) = 0   THEN 'deposit_type'
            ELSE 'repeat_guest' END                                     AS dimension,
       COALESCE(market_segment, deposit_type,
                CASE is_repeated_guest WHEN 1 THEN 'Repeat guest' ELSE 'First-time guest' END) AS value,
       COUNT(*)                                                         AS bookings,
       ROUND(100.0 * AVG(is_canceled), 1)                               AS cancel_rate_pct,
       ROUND(SUM(CASE WHEN is_canceled = 1 THEN room_revenue END), 0)   AS revenue_lost
FROM bookings
GROUP BY GROUPING SETS ((market_segment), (deposit_type), (is_repeated_guest))
HAVING COUNT(*) >= 100
ORDER BY dimension, cancel_rate_pct DESC;
