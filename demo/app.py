"""
FRED Forecaster - Streamlit Demo App

This app demonstrates the use of the fred_forecaster package for generating
time series forecasts of FRED economic data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import arviz as az

from fred_forecaster import (
    fetch_fred_data,
    fit_sarimax_model, 
    generate_simulations,
    fit_bayesian_model,
    generate_bayesian_simulations,
    calibrate_simulations,
    plot_forecasts, 
    plot_drop_probabilities
)
from fred_forecaster.data import get_series_name, get_series_title


def main():
    st.set_page_config(
        page_title="FRED Forecaster Demo",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("FRED Series Forecasting Demo")
    st.sidebar.image("https://fred.stlouisfed.org/images/masthead-88.png", width=200)
    
    st.sidebar.markdown("## Configuration")

    # 1. User input: FRED Series ID
    fred_series_id = st.sidebar.text_input(
        "FRED Series ID:",
        value="GFDEBTN",
        help="Example: GFDEBTN for total public debt, BOGZ1FL192090005Q for household net worth."
    )
    
    # 2. User input: Model selection
    model_type = st.sidebar.radio(
        "Select Forecasting Model:",
        ["SARIMAX (Classical)", "Bayesian Structural Time Series"],
        index=0,
        help="SARIMAX is faster but less robust. Bayesian model provides more insight into uncertainty."
    )

    # 3. User input: Use calibration or not
    calibration_toggle = st.sidebar.checkbox(
        "Calibrate to CBO targets?", value=True,
        help="Reweight simulations to match Congressional Budget Office forecasts"
    )
    
    # 4. Advanced options
    with st.sidebar.expander("Advanced Options"):
        num_simulations = st.number_input(
            "Number of simulations", 
            min_value=100, 
            max_value=10000, 
            value=1000,
            step=100
        )
        
        show_paths = st.number_input(
            "Number of simulation paths to show",
            min_value=0,
            max_value=200,
            value=50,
            step=10
        )
        
        forecast_end = st.text_input(
            "Forecast end date (YYYYQN)",
            value="2028Q4"
        )

    # Run the forecast
    if st.button("Load Data and Run Forecast", type="primary"):
        try:
            with st.spinner("Fetching data from FRED..."):
                df_quarterly = fetch_fred_data(fred_series_id)
                series_name = get_series_name(df_quarterly)
                series_title = get_series_title(df_quarterly)
                
            # Show the historical data
            st.subheader(f"Historical Data: {series_title}")
            st.dataframe(df_quarterly.style.format({series_name: "{:.2f}"}))
            
            # Run selected model
            sim_array = None
            forecast_index = None
            
            if model_type == "SARIMAX (Classical)":
                # Classical SARIMAX approach
                with st.spinner("Fitting SARIMAX model..."):
                    results = fit_sarimax_model(df_quarterly)
                    
                with st.spinner("Generating simulations..."):
                    sim_array, forecast_index = generate_simulations(
                        results, 
                        df_quarterly,
                        end=forecast_end,
                        N=num_simulations
                    )
                    
            else:
                # Bayesian approach
                with st.spinner("Fitting Bayesian model (this may take a few minutes)..."):
                    try:
                        model, idata = fit_bayesian_model(df_quarterly)
                        
                        # Create Bayesian diagnostics in a collapsible section
                        with st.expander("Bayesian Model Diagnostics", expanded=False):
                            st.write("Posterior distributions of key parameters:")
                            
                            # Create diagnostic plots using Arviz
                            param_names = ["sigma_level", "sigma_trend", "sigma_seasonal", "sigma_obs"]
                            for param in param_names:
                                trace_plot = az.plot_trace(idata, var_names=[param])
                                st.pyplot(trace_plot[0][0].figure)
                    
                    except Exception as e:
                        st.error(f"Error fitting Bayesian model: {str(e)}")
                        st.info("Falling back to SARIMAX model due to Bayesian model error.")
                        
                        # Fallback to SARIMAX
                        with st.spinner("Fitting SARIMAX model instead..."):
                            results = fit_sarimax_model(df_quarterly)
                            
                        with st.spinner("Generating simulations..."):
                            sim_array, forecast_index = generate_simulations(
                                results, 
                                df_quarterly,
                                end=forecast_end,
                                N=num_simulations
                            )
                    else:
                        # If Bayesian model succeeded, generate simulations
                        with st.spinner("Generating Bayesian simulations..."):
                            sim_array, forecast_index = generate_bayesian_simulations(
                                model, 
                                idata, 
                                df_quarterly,
                                end=forecast_end,
                                N=num_simulations
                            )

            # Optional: Calibration
            weights = None
            if calibration_toggle:
                try:
                    with st.spinner("Calibrating simulations to CBO targets..."):
                        weights = calibrate_simulations(sim_array, forecast_index)
                except Exception as e:
                    st.warning(f"Calibration failed: {str(e)}. Proceeding without calibration.")

            # Visualization
            st.subheader("Forecast Results")
            
            # Create columns for forecast and probabilities
            col1, col2 = st.columns(2)
            
            # Plot forecasts
            fig_forecasts = plot_forecasts(
                df_quarterly, 
                sim_array, 
                forecast_index, 
                weights, 
                num_paths_to_show=show_paths
            )
            col1.plotly_chart(fig_forecasts, use_container_width=True)

            # Probability of drop
            fig_drop_prob = plot_drop_probabilities(sim_array, forecast_index, weights)
            col2.plotly_chart(fig_drop_prob, use_container_width=True)
            
            # Bayesian insights (only for Bayesian model)
            if model_type == "Bayesian Structural Time Series" and 'model' in locals() and 'idata' in locals():
                with st.expander("Bayesian Model Component Decomposition", expanded=False):
                    st.write("""
                    The Bayesian structural time series model decomposes the time series into:
                    - Level: The base value of the series
                    - Trend: The directional component
                    - Seasonality: Quarterly patterns in the data
                    """)
                    
                    # Create time component plots with Plotly
                    components_fig = go.Figure()
                    
                    # Convert index to timestamp for plotting
                    plot_dates = df_quarterly.index.to_timestamp()
                    
                    # Get component means
                    level_mean = idata.posterior["level"].mean(["chain", "draw"]).values
                    trend_mean = idata.posterior["trend"].mean(["chain", "draw"]).values
                    seasonal_mean = idata.posterior["seasonal"].mean(["chain", "draw"]).values
                    
                    # Add traces
                    components_fig.add_trace(go.Scatter(
                        x=plot_dates, y=level_mean, mode='lines', name='Level Component',
                        line=dict(color='blue')
                    ))
                    
                    components_fig.add_trace(go.Scatter(
                        x=plot_dates, y=trend_mean, mode='lines', name='Trend Component',
                        line=dict(color='red')
                    ))
                    
                    components_fig.add_trace(go.Scatter(
                        x=plot_dates, y=seasonal_mean, mode='lines', name='Seasonal Component',
                        line=dict(color='green')
                    ))
                    
                    # Update layout
                    components_fig.update_layout(
                        title="Time Series Components",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(components_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.markdown(
        "This demo app showcases the fred_forecaster package for time series "
        "forecasting with FRED economic data using both classical SARIMAX and "
        "Bayesian approaches."
    )
    st.sidebar.markdown(
        "For more information on FRED series, visit "
        "[FRED Economic Data](https://fred.stlouisfed.org/)."
    )
    st.sidebar.markdown("© 2025 fred_forecaster")


if __name__ == "__main__":
    main()