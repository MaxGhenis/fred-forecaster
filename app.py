import streamlit as st
import pandas as pd
from src.data import fetch_fred_data
from src.forecast import fit_sarimax_model, generate_simulations
from src.calibration import calibrate_simulations
from src.visualization import plot_forecasts, plot_drop_probabilities


def main():
    st.title("FRED Series Forecasting App")

    # 1. User input: FRED Series ID
    fred_series_id = st.text_input(
        "Enter FRED Series ID:",
        value="GFDEBTN",
        help="Example: GFDEBTN for total public debt.",
    )

    # 2. User input: Use calibration or not
    calibration_toggle = st.checkbox(
        "Calibrate to CBO (hard-coded) targets?", value=True
    )

    # 3. Fetch data
    if st.button("Load Data and Run Forecast"):
        with st.spinner("Fetching data from FRED..."):
            df_quarterly = fetch_fred_data(fred_series_id)

        # 4. Fit SARIMAX
        with st.spinner("Fitting SARIMAX model..."):
            results = fit_sarimax_model(df_quarterly["Debt"])

        # 5. Generate simulations
        with st.spinner("Generating simulations..."):
            sim_array, forecast_index = generate_simulations(results, df_quarterly)

        # 6. Optional: Calibration
        weights = None
        if calibration_toggle:
            with st.spinner("Calibrating simulations to CBO targets..."):
                weights = calibrate_simulations(sim_array, forecast_index)

        # 7. Visualization
        st.subheader("Forecast Results")
        fig_forecasts = plot_forecasts(df_quarterly, sim_array, forecast_index, weights)
        st.pyplot(fig_forecasts)

        # 8. Probability of drop
        fig_drop_prob = plot_drop_probabilities(sim_array, forecast_index, weights)
        st.pyplot(fig_drop_prob)

    st.markdown("---")
    st.markdown("© 2025 Your Organization")


if __name__ == "__main__":
    main()
