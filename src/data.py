import pandas as pd
import numpy as np
import os
from fredapi import Fred


def fetch_fred_data(series_id: str) -> pd.DataFrame:
    """
    Fetches a FRED series by ID, returns a quarterly PeriodIndex DataFrame.
    Expects an environment variable FRED_API_KEY or similar.
    """
    api_key = os.getenv("FRED_API_KEY", None)
    if not api_key:
        raise ValueError("FRED_API_KEY not set in environment.")

    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id).to_frame(name="Debt_millions")
    series.index.name = "Date"

    # Convert to trillions, convert index to quarterly PeriodIndex
    series["Debt"] = series["Debt_millions"] / 1e6
    series.drop(columns=["Debt_millions"], inplace=True)
    series.index = pd.to_datetime(series.index)

    df_quarterly = series.resample("QE", origin="end").last()
    df_quarterly.index = df_quarterly.index.to_period("Q-DEC").sort_values()

    return df_quarterly
