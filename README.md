# FRED Series Forecasting App

This repository contains a **Streamlit** application that downloads historical data from the Federal Reserve Economic Data (**FRED**), fits a SARIMAX time-series model, generates stochastic forecasts, and (optionally) reweights those forecasts to match hard-coded **CBO** (Congressional Budget Office) target values. The application also displays probabilities of quarter-over-quarter declines in the forecasted metric.

## Features

- **Flexible FRED data retrieval**: Users can select any FRED series ID (e.g., _GFDEBTN_ for total public debt).
- **SARIMAX modeling**: The model automatically fits a SARIMAX(1,1,1)x(0,1,0)[4] to quarterly data.
- **Simulation**: Generates multiple (e.g., 1,000) random simulations for future quarters.
- **Optional calibration**: Reweights simulations so that specific quarter/year forecast points match user-defined targets (e.g., CBO Q4 forecasts).
- **Probability analysis**: Calculates the probability of a quarter-over-quarter decline in the forecasted series, including overall probabilities and specific windows.
- **Interactive visualizations**: Plots forecast distributions, percentile bounds, weighted means, and bar charts of drop probabilities.

## Repository Structure

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── data.py
    ├── forecast.py
    ├── calibration.py
    ├── visualization.py
    ├── constants.py
    └── __init__.py
```

**Brief Description**

- `app.py`: Main Streamlit app, handles user interactions and ties everything together.
- `requirements.txt`: Lists dependencies needed to run this app (e.g., `streamlit`, `pandas`, `numpy`, etc.).
- `src/data.py`: Contains functions for fetching and preprocessing FRED data.
- `src/forecast.py`: Fits SARIMAX models and generates simulation paths.
- `src/calibration.py`: Implements optional reweighting to match external (e.g., CBO) targets.
- `src/visualization.py`: Utility functions to create and return Matplotlib figures for Streamlit to display.
- `src/constants.py` (optional): Central place for storing references like CBO targets or other repeated data.

## Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/your-username/streamlit_app.git
cd streamlit_app
```

### 2. Install dependencies

We recommend creating a virtual environment, then installing packages:

```bash
pip install -r requirements.txt
```

This will ensure that `streamlit`, `fredapi`, `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`, and other dependencies are installed.

### 3. Set your FRED API key

The app requires a FRED API key for `fredapi`. You can request an API key at:  
[https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

Set the environment variable `FRED_API_KEY` to your key. For example, on Linux or macOS:

```bash
export FRED_API_KEY="YOUR_FRED_API_KEY_HERE"
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

This will launch a local web server. By default, a browser tab should open at [http://localhost:8501](http://localhost:8501) (or similar).

## Usage

1. **Enter a FRED series ID** in the text box (defaults to `GFDEBTN`, total public debt).
2. **Toggle calibration** if you want to force certain quarters (like Q4 of each year) to match hard-coded CBO forecasts.
3. **Click "Load Data and Run Forecast"** to fetch data, fit the model, and generate simulations.
4. **Review the plots**:
   - A line chart of simulations (grey lines), the historical data (black line), and if calibration is on, the weighted mean plus percentile bounds.
   - A bar chart of quarter-over-quarter drop probabilities from a certain future quarter onward.
   - Text indicators of overall drop probabilities.

## Customization

- **Model parameters**: Update SARIMAX orders in `src/forecast.py` to suit your data.
- **Forecast horizon**: Change the end period in `generate_simulations` if you want to forecast beyond 2028Q4 or pick a shorter horizon.
- **Calibration**: Modify `src/calibration.py` to change which quarters or which external targets you wish to reweight to.
- **Visualization**: Enhance or change plots in `src/visualization.py` to better suit your needs or incorporate additional interactive charts (e.g., Plotly, Altair).

## Contributing

If you’d like to contribute:

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.
