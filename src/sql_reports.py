"""Run every query in sql/ against the clean Parquet data with DuckDB, and save each result as CSV."""
import duckdb

from src import config


def run() -> dict:
    con = duckdb.connect()
    path = (config.DATA_CLEAN / "bookings.parquet").as_posix()
    con.execute(f"CREATE VIEW bookings AS SELECT * FROM read_parquet('{path}')")
    results = {}
    for f in sorted(config.SQL_DIR.glob("*.sql")):
        df = con.execute(f.read_text()).df()
        df.to_csv(config.POWERBI / f"sql_{f.stem}.csv", index=False)
        results[f.stem] = df
    return results


if __name__ == "__main__":
    for k, v in run().items():
        print(f"\n== {k} ({len(v)} rows)\n{v.head(12).to_string()}")
