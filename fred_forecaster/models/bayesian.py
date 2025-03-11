"""Bayesian time series forecasting models."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Tuple, Any, Union


def fit_bayesian_model(ts_data: Union[pd.Series, pd.DataFrame]):
    """
    Fits a Bayesian structural time series model to the provided data.
    Uses PyMC for Bayesian inference.
    
    Parameters
    ----------
    ts_data : Union[pd.Series, pd.DataFrame]
        Time series data to fit. If DataFrame, the first column is used.
        
    Returns
    -------
    model : pm.Model
        PyMC model object
    idata : az.InferenceData
        Inference data containing posterior samples
    """
    # Convert DataFrame to Series if needed
    if isinstance(ts_data, pd.DataFrame):
        ts_data = ts_data.iloc[:, 0]
        
    # Convert to numpy array for modeling
    y = ts_data.values
    n = len(y)
    
    # Build PyMC model
    with pm.Model() as model:
        # Standard deviation priors for the different components
        sigma_level = pm.HalfNormal("sigma_level", sigma=0.1)
        sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.01)
        sigma_seasonal = pm.HalfNormal("sigma_seasonal", sigma=0.01)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)
        
        # Initial values using dist() API to avoid registration errors
        init_level_dist = pm.Normal.dist(mu=y[0], sigma=1)
        init_trend_dist = pm.Normal.dist(mu=0, sigma=0.1)
        init_seasonal_dist = pm.Normal.dist(mu=0, sigma=0.1, shape=4)
        
        # Level and trend components (local linear trend model)
        level = pm.GaussianRandomWalk(
            "level", 
            sigma=sigma_level, 
            init_dist=init_level_dist,
            shape=n
        )
        trend = pm.GaussianRandomWalk(
            "trend", 
            sigma=sigma_trend, 
            init_dist=init_trend_dist,
            shape=n
        )
        
        # Seasonal component (quarterly seasonality)
        period = 4  # quarterly data
        seasonal = pm.GaussianRandomWalk(
            "seasonal", 
            sigma=sigma_seasonal,
            init_dist=init_seasonal_dist,
            shape=n
        )
        
        # Expected value
        mu = level + trend + seasonal
        
        # Observations
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)
        
        # Inference - use a smaller sample for faster results
        idata = pm.sample(500, tune=500, chains=2, return_inferencedata=True)
    
    return model, idata


def generate_bayesian_simulations(
    model: Any, 
    idata: az.InferenceData, 
    df_quarterly: pd.DataFrame, 
    end: str = "2028Q4", 
    N: int = 1000
) -> Tuple[np.ndarray, pd.PeriodIndex]:
    """
    Generate N random simulations from the fitted Bayesian model,
    forecasting until the given 'end' Period (e.g., 2028Q4).

    Parameters
    ----------
    model : pm.Model
        Fitted PyMC model
    idata : az.InferenceData
        Inference data from the model
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
    
    n_data = len(df_quarterly)
    # Get the values from the first column
    y = df_quarterly.iloc[:, 0].values
    
    # Setup forecast model
    with model:
        # Get parameter posterior samples
        level_trace = idata.posterior["level"].values
        trend_trace = idata.posterior["trend"].values
        seasonal_trace = idata.posterior["seasonal"].values
        sigma_obs_trace = idata.posterior["sigma_obs"].values
        
        # Flatten chains
        level_samples = level_trace.reshape(-1, n_data)
        trend_samples = trend_trace.reshape(-1, n_data)
        seasonal_samples = seasonal_trace.reshape(-1, n_data)
        sigma_samples = sigma_obs_trace.flatten()
        
        # Generate forecasts
        np.random.seed(42)
        sim_array = np.zeros((steps, N))
        
        # Generate N different forecast paths
        for i in range(N):
            # Randomly select a posterior sample
            idx = np.random.randint(0, len(sigma_samples))
            
            # Get last values from the model
            last_level = level_samples[idx, -1]
            last_trend = trend_samples[idx, -1]
            season_pattern = seasonal_samples[idx, -4:]  # Last year's seasonality
            sigma = sigma_samples[idx]
            
            # Forecast values
            forecast = np.zeros(steps)
            for j in range(steps):
                # Add in trend component with some noise
                level_next = last_level + last_trend + np.random.normal(0, sigma/10)
                trend_next = last_trend + np.random.normal(0, sigma/20)
                
                # Add in seasonal component
                season_idx = j % 4
                seasonal_component = season_pattern[season_idx] + np.random.normal(0, sigma/20)
                
                # Combine components
                forecast[j] = level_next + seasonal_component + np.random.normal(0, sigma)
                
                # Update for next step
                last_level = level_next
                last_trend = trend_next
            
            sim_array[:, i] = forecast
    
    forecast_index = pd.period_range(start_forecast, periods=steps, freq="Q-DEC")
    return sim_array, forecast_index