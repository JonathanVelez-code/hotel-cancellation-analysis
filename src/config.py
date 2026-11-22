"""Paths and settings."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
POWERBI = OUTPUTS / "powerbi"
SQL_DIR = ROOT / "sql"

# Public copy of the Hotel Booking Demand dataset (António, Almeida & Nunes, Data in Brief, 2019)
DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
RAW_FILE = DATA_RAW / "hotels.csv"

# Train on arrivals before this date, test on arrivals from this date on (Jan-Aug 2017)
TEST_START = pd.Timestamp("2017-01-01")
SEED = 42

for p in (DATA_RAW, DATA_CLEAN, OUTPUTS, FIGURES, POWERBI):
    p.mkdir(parents=True, exist_ok=True)
