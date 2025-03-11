# fred_forecaster

A Python package for time series forecasting of Federal Reserve Economic Data (FRED).

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Flexible FRED data retrieval**: Fetch any FRED series using its ID (e.g., _GFDEBTN_ for total public debt).
- **Multiple forecasting approaches**:
  - **Classical SARIMAX modeling**: Fit SARIMAX(1,1,1)x(0,1,0)[4] to quarterly data.
  - **Bayesian structural time series**: Decompose data into level, trend, and seasonal components using PyMC.
- **Simulation**: Generate multiple (e.g., 1,000) random simulations for future quarters.
- **Calibration**: Reweight simulations to match external targets (e.g., CBO forecasts).
- **Probability analysis**: Calculate probabilities of quarter-over-quarter declines.
- **Visualization**: Create plots of forecasts, probability distributions, and model components.

## Installation

### From PyPI

```bash
pip install fred-forecaster
```

### From source

```bash
git clone https://github.com/maxghenis/fred-forecaster.git
cd fred-forecaster
pip install -e .  # Install core package
pip install -e ".[dev]"  # Include development dependencies
pip install -e ".[app]"  # Include Streamlit app dependencies
```

## Usage

### Basic usage

```python
import os
from fred_forecaster import (
    fetch_fred_data,
    fit_sarimax_model,
    generate_simulations,
    plot_forecasts
)

# Set your FRED API key
os.environ["FRED_API_KEY"] = "your_api_key_here"

# Fetch FRED data
data = fetch_fred_data("GFDEBTN")  # Federal debt

# Fit SARIMAX model
model = fit_sarimax_model(data["Debt"])

# Generate simulations
simulations, forecast_index = generate_simulations(model, data, end="2028Q4", N=1000)

# Plot results
fig = plot_forecasts(data, simulations, forecast_index)
fig.savefig("forecast.png")
```

### Bayesian forecasting

```python
from fred_forecaster import (
    fetch_fred_data,
    fit_bayesian_model,
    generate_bayesian_simulations,
    plot_forecasts
)

# Fetch data
data = fetch_fred_data("GFDEBTN")

# Fit Bayesian model
model, idata = fit_bayesian_model(data["Debt"])

# Generate simulations
simulations, forecast_index = generate_bayesian_simulations(
    model, idata, data, end="2028Q4", N=1000
)

# Plot results
fig = plot_forecasts(data, simulations, forecast_index)
```

### Calibration to external targets

```python
from fred_forecaster import calibrate_simulations

# Define your own targets (or use the default CBO targets)
targets = {2024: 35.230, 2025: 37.209}

# Calibrate simulations
weights = calibrate_simulations(simulations, forecast_index, targets)

# Plot calibrated results
fig = plot_forecasts(data, simulations, forecast_index, weights)
```

## Demo App

The package includes a Streamlit demo app that showcases its functionality:

```bash
cd demo
streamlit run app.py
```

This will open a web browser with an interactive interface where you can:
- Select a FRED series ID
- Choose between SARIMAX and Bayesian models
- Toggle calibration to CBO targets
- Adjust simulation parameters
- View forecasts and probability analyses

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running tests

```bash
pytest  # Run regular tests
pytest --run-slow  # Include slow tests (e.g., Bayesian model tests)
pytest --cov=fred_forecaster  # Run with coverage
```

## License

MIT

## Acknowledgments

- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- [PyMC](https://www.pymc.io/)
- [statsmodels](https://www.statsmodels.org/)
- [Streamlit](https://streamlit.io/)