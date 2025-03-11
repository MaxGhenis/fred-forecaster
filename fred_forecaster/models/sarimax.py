"""SARIMAX time series forecasting models."""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Union


def fit_sarimax_model(ts_data: Union[pd.Series, pd.DataFrame]):
    """
    Fits a SARIMAX model to the provided time series data.
    
    Parameters
    ----------
    ts_data : Union[pd.Series, pd.DataFrame]
        Time series data to fit. If DataFrame, the first column is used.
        
    Returns
    -------
    SARIMAXResults
        Fitted SARIMAX model
    """
    # Convert DataFrame to Series if needed
    if isinstance(ts_data, pd.DataFrame):
        ts_data = ts_data.iloc[:, 0]
        
    p, d, q = 1, 1, 1
    P, D, Q, m = 0, 1, 0, 4  # Example
    model = SARIMAX(
        ts_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results


def generate_simulations(
    results, df_quarterly: pd.DataFrame, end: str = "2028Q4", N: int = 1000
) -> Tuple[np.ndarray, pd.PeriodIndex]:
    """
    Generate N random simulations from the fitted SARIMAX results,
    forecasting until the given 'end' Period (e.g., 2028Q4).

    Parameters
    ----------
    results : SARIMAXResults
        Fitted SARIMAX model results
    df_quarterly : pd.DataFrame
        Historical data with PeriodIndex
    end : str
        End period for forecast in format 'YYYYQN' (e.g., '2028Q4')
    N : int
        Number of simulations to generate

    Returns
    -------
    sim_array : np.ndarray
        Shape (steps, N), each column is one simulation path.
    forecast_index : pd.PeriodIndex
        The quarters covered by the forecast.
    """
    # Forecast range
    last_period = df_quarterly.index[-1]
    # Create the next period after last_period correctly
    start_forecast = pd.Period(f"{last_period.year}Q{last_period.quarter}", freq="Q-DEC")
    if last_period.quarter < 4:
        start_forecast = pd.Period(f"{last_period.year}Q{last_period.quarter + 1}", freq="Q-DEC")
    else:
        start_forecast = pd.Period(f"{last_period.year + 1}Q1", freq="Q-DEC")
    end_forecast = pd.Period(end, freq="Q-DEC")

    def quarter_index(prd: pd.Period) -> int:
        return prd.year * 4 + prd.quarter

    steps = quarter_index(end_forecast) - quarter_index(start_forecast) + 1
    if steps < 1:
        raise ValueError(
            f"Invalid forecast horizon: {start_forecast} to {end_forecast}"
        )

    # Simulate
    np.random.seed(42)
    sim_array = results.simulate(nsimulations=steps, repetitions=N, anchor="end")

    # Some versions give shape (N, steps), ensure shape is (steps, N).
    if sim_array.shape[0] == N:
        sim_array = sim_array.T

    if hasattr(sim_array, "values"):
        sim_array = sim_array.values
    else:
        sim_array = np.asarray(sim_array)

    forecast_index = pd.period_range(start_forecast, periods=steps, freq="Q-DEC")
    return sim_array, forecast_index