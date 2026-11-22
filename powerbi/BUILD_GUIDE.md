# Power BI dashboard: build guide

Run `python -m src.pipeline`, open Power BI Desktop, then go to **Get data → Text/CSV** and load the files in `outputs/powerbi/`.

## Model
- `fact_bookings[arrival_date]` → `dim_date[date]` (many-to-one). Then mark `dim_date` as the date table.
- The other tables (`sql_*`, `weekly_overbooking`, `gains_curve`, `feature_importance`) can stand alone.

## Measures (in `fact_bookings`)
```DAX
Bookings = COUNTROWS ( fact_bookings )
Cancellations = SUM ( fact_bookings[is_canceled] )
Cancel Rate = DIVIDE ( [Cancellations], [Bookings] )
Realized Revenue = CALCULATE ( SUM ( fact_bookings[room_revenue] ), fact_bookings[is_canceled] = 0 )
Revenue Lost = CALCULATE ( SUM ( fact_bookings[room_revenue] ), fact_bookings[is_canceled] = 1 )
Pct Revenue Lost = DIVIDE ( [Revenue Lost], [Realized Revenue] + [Revenue Lost] )
Avg Daily Rate = CALCULATE ( AVERAGE ( fact_bookings[adr] ), fact_bookings[is_canceled] = 0 )
Avg Predicted Risk = AVERAGE ( fact_bookings[cancel_risk] )
```

## Pages
1. **Cancellation overview**
   - Cards: Bookings, Cancel Rate, Revenue Lost, Pct Revenue Lost
   - Column chart: cancel rate by `lead_time_bucket` (sort by days)
   - Bar chart: cancel rate by `market_segment`
   - Slicers: hotel, year
2. **Seasonality**
   - Line chart: Cancel Rate and Avg Daily Rate by `dim_date[year_month]`, per hotel
3. **Risk & targeting**
   - Line chart from `gains_curve`: % of bookings targeted vs % of cancellations captured, by model
   - Table of high-risk bookings (`risk_band` = High) with market segment, lead time and deposit type
4. **Overbooking**
   - Line chart from `weekly_overbooking`: actual vs expected cancellations per week, plus `safe_overbook_rooms`, with a hotel slicer

Save as `powerbi/hotel_cancellations.pbix`, take screenshots, and add them to the README.
