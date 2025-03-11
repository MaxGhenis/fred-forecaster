"""Functions for fetching and preprocessing FRED data."""

import pandas as pd
import numpy as np
import os
from fredapi import Fred
from typing import Optional, Dict, Any


def fetch_fred_data(
    series_id: str, 
    api_key: Optional[str] = None,
    value_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches a FRED series by ID, returns a quarterly PeriodIndex DataFrame.
    
    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'GFDEBTN' for total public debt)
    api_key : str, optional
        FRED API key. If None, will attempt to read from FRED_API_KEY environment variable
    value_name : str, optional
        Name to use for the value column. If None, uses the series ID.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with PeriodIndex and value column
        
    Raises
    ------
    ValueError
        If FRED_API_KEY is not set and api_key is not provided
    """
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY", None)
        if not api_key:
            raise ValueError("FRED_API_KEY not set in environment.")

    # Get series metadata to determine name and units
    fred = Fred(api_key=api_key)
    series_info = fred.get_series_info(series_id)
    
    # Get actual data
    series_data = fred.get_series(series_id)
    
    # Determine column name
    if value_name is None:
        value_name = series_id
    
    # Create DataFrame
    series = series_data.to_frame(name=value_name)
    series.index.name = "Date"
    series.index = pd.to_datetime(series.index)
    
    # Convert to quarterly
    df_quarterly = series.resample("QE", origin="end").last()
    df_quarterly.index = df_quarterly.index.to_period("Q-DEC").sort_values()
    
    # Add metadata as attributes
    df_quarterly.attrs["title"] = series_info.get("title", value_name)
    df_quarterly.attrs["units"] = series_info.get("units", "")
    df_quarterly.attrs["series_id"] = series_id
    df_quarterly.attrs["frequency"] = series_info.get("frequency", "Quarterly")

    return df_quarterly


def get_series_name(df: pd.DataFrame) -> str:
    """
    Get the name of the value column from a DataFrame returned by fetch_fred_data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by fetch_fred_data
        
    Returns
    -------
    str
        Name of the value column
    """
    # Return the first column name that's not an index
    return df.columns[0]


def get_series_title(df: pd.DataFrame) -> str:
    """
    Get the title of the series from a DataFrame returned by fetch_fred_data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by fetch_fred_data
        
    Returns
    -------
    str
        Title of the series
    """
    return df.attrs.get("title", get_series_name(df))