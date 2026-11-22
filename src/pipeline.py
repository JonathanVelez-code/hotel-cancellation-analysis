"""Run everything: download + clean -> SQL analysis -> model -> business analysis -> Power BI tables -> charts.

    python -m src.pipeline
"""
import json
import logging
import time

from src import business, config, data, export_powerbi, figures, model, sql_reports


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pipeline")
    start = time.time()

    d, report = data.run()
    log.info("Clean data: %s bookings (%s removed)", f"{report['clean_rows']:,}", report["rows_removed"])

    q = sql_reports.run()
    log.info("SQL reports: %s", ", ".join(q))

    m = model.run(d)
    for name, r in m["summary"]["models"].items():
        log.info("%s: ROC AUC %.3f, PR AUC %.3f", name, r["roc_auc"], r["pr_auc"])

    b = business.run(m["test"], m["train"])
    t20 = b["risk_targeting_xgboost"]["top_20pct"]
    log.info("Top 20%% riskiest bookings hold %s%% of cancellations", t20["share_of_all_cancellations_pct"])
    log.info("Weekly cancellation forecast WAPE: model %s%% vs baseline %s%%",
             b["overbooking"]["weekly_cancellation_forecast_wape_pct_model"],
             b["overbooking"]["weekly_cancellation_forecast_wape_pct_baseline"])

    export_powerbi.run(d, m["test"])
    figures.run(m["test"])

    summary = {"runtime_seconds": round(time.time() - start, 1), "clean_bookings": report["clean_rows"],
               "xgboost_roc_auc": m["summary"]["models"]["XGBoost"]["roc_auc"],
               "top20_share_of_cancellations_pct": t20["share_of_all_cancellations_pct"]}
    (config.OUTPUTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done in %.1fs", summary["runtime_seconds"])


if __name__ == "__main__":
    main()
