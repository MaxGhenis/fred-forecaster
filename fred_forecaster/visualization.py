"""Visualization functions for forecast results."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Tuple, Dict, Any, List

from .data import get_series_name, get_series_title


def plot_forecasts(
    df_quarterly: pd.DataFrame, 
    sim_array: np.ndarray, 
    forecast_index: pd.PeriodIndex, 
    weights: Optional[np.ndarray] = None,
    num_paths_to_show: int = 50
) -> go.Figure:
    """
    Plot historical data + simulation paths + (optional) weighted means using Plotly.
    
    Parameters
    ----------
    df_quarterly : pd.DataFrame
        Historical data with PeriodIndex
    sim_array : np.ndarray
        Array of shape (steps, N) containing N simulation paths
    forecast_index : pd.PeriodIndex
        Index of time periods corresponding to sim_array rows
    weights : np.ndarray, optional
        Weight vector of length N. If None, equal weights are used.
    num_paths_to_show : int, optional
        Number of individual simulation paths to show (default: 50)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Get series name and title
    series_name = get_series_name(df_quarterly)
    series_title = get_series_title(df_quarterly)
    units = df_quarterly.attrs.get("units", "")
    
    # Create figure
    fig = go.Figure()
    
    # Convert historical index to timestamps for plotting
    hist_dates = df_quarterly.index.to_timestamp()
    
    # Convert forecast index to timestamps
    forecast_dates = forecast_index.to_timestamp()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=df_quarterly[series_name],
            mode="lines",
            line=dict(color="black", width=2),
            name="Historical"
        )
    )
    
    # Plot a subset of individual simulation paths
    if num_paths_to_show > 0:
        paths_to_show = min(num_paths_to_show, sim_array.shape[1])
        indices = np.random.choice(sim_array.shape[1], paths_to_show, replace=False)
        
        for idx in indices:
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=sim_array[:, idx],
                    mode="lines",
                    line=dict(color="rgba(200, 200, 200, 0.3)"),
                    showlegend=False
                )
            )
    
    # Weighted mean or unweighted average
    if weights is not None:
        # Calculate weighted mean
        weighted_mean = np.dot(sim_array, weights)
        
        # Calculate percentiles
        lower_bound = np.percentile(sim_array, 5, axis=1)
        upper_bound = np.percentile(sim_array, 95, axis=1)
        
        # Add weighted mean
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=weighted_mean,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Weighted Mean"
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",
                line=dict(color="rgba(0, 0, 255, 0)"),
                name="5-95% Confidence Interval"
            )
        )
    else:
        # Calculate simple percentiles
        lower_bound = np.percentile(sim_array, 5, axis=1)
        upper_bound = np.percentile(sim_array, 95, axis=1)
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",
                line=dict(color="rgba(255, 165, 0, 0)"),
                name="5-95% Confidence Interval"
            )
        )
    
    # Update layout
    y_axis_title = f"{series_title}"
    if units:
        y_axis_title += f" ({units})"
        
    fig.update_layout(
        title=f"{series_title} Forecast",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Add vertical line to separate historical from forecast
    last_historical_date = hist_dates[-1]
    # Convert timestamp to string format for plotly
    vline_date = last_historical_date.strftime('%Y-%m-%d')
    
    # Add a shape instead of using add_vline
    fig.add_shape(
        type="line",
        x0=vline_date,
        x1=vline_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    # Add annotation for the forecast start
    fig.add_annotation(
        x=vline_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        textangle=0,
        xanchor="right",
        yanchor="top",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
    )
    
    return fig


def plot_drop_probabilities(
    sim_array: np.ndarray, 
    forecast_index: pd.PeriodIndex, 
    weights: Optional[np.ndarray] = None,
    start_year: int = 2025
) -> go.Figure:
    """
    Plot bar chart for quarter-over-quarter drop probabilities using Plotly.
    
    Parameters
    ----------
    sim_array : np.ndarray
        Array of shape (steps, N) containing N simulation paths
    forecast_index : pd.PeriodIndex
        Index of time periods corresponding to sim_array rows
    weights : np.ndarray, optional
        Weight vector of length N. If None, equal weights are used.
    start_year : int, optional
        Year to start computing drop probabilities from (default: 2025)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # If no weights, assume uniform
    N = sim_array.shape[1]
    if weights is None:
        weights = np.ones(N) / N

    # Probability for each quarter from start_year onward
    prob_fall_data = []
    for i in range(1, len(forecast_index)):
        if forecast_index[i].year < start_year:
            continue
        arr_this = sim_array[i, :]
        arr_prev = sim_array[i - 1, :]
        prob = np.dot(weights, (arr_this < arr_prev).astype(float))
        prob_fall_data.append((forecast_index[i], prob))

    df_prob_fall = pd.DataFrame(
        prob_fall_data, columns=["Quarter", "ProbDecrease"]
    ).set_index("Quarter")

    # Overall probability of at least one drop
    diffs = np.diff(sim_array, axis=0)
    has_decline = (diffs < 0).any(axis=0).astype(float)
    overall_prob_drop = np.dot(weights, has_decline)

    # Probability from start_year onward
    start_idx = None
    for i, q in enumerate(forecast_index):
        if q.year >= start_year:
            start_idx = i
            break
    if start_idx is not None:
        relevant_diffs = np.diff(sim_array[start_idx:], axis=0)
        has_decline_start_year = (relevant_diffs < 0).any(axis=0).astype(float)
        prob_drop_start_year_on = np.dot(weights, has_decline_start_year)
    else:
        prob_drop_start_year_on = np.nan

    # Create plot
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=df_prob_fall.index.astype(str),
            y=df_prob_fall["ProbDecrease"],
            marker_color="orange"
        )
    )
    
    # Add text annotation with overall probabilities
    fig.add_annotation(
        x=0.01,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Overall: {overall_prob_drop:.2%}<br>{start_year}+ window: {prob_drop_start_year_on:.2%}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font=dict(size=12)
    )
    
    # Update layout
    fig.update_layout(
        title=f"Quarter-over-Quarter Decrease Probabilities ({start_year}Q1+)",
        xaxis_title="Quarter",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white"
    )
    
    return fig