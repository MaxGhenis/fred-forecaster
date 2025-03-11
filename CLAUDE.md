# Fred Forecaster Development Guide

## Commands

### Development
- **Install package in dev mode**: `pip install -e .`
- **Install with app dependencies**: `pip install -e ".[app]"`
- **Install with dev dependencies**: `pip install -e ".[dev]"`
- **Run demo app**: `cd demo && streamlit run app.py`
- **Environment setup**: Set `FRED_API_KEY` environment variable
- **Format code**: `black .`
- **Check types**: `mypy fred_forecaster`
- **Lint code**: `flake8 fred_forecaster`

### Testing
- **Run tests**: `pytest tests/`
- **Run specific test**: `pytest tests/test_data.py::TestData::test_fetch_fred_data`
- **Run tests with coverage**: `pytest --cov=fred_forecaster tests/`
- **Run slow tests**: `pytest --run-slow tests/`

### Packaging and Distribution
- **Build package**: `python -m build`
- **Check package**: `twine check dist/*`
- **Upload to PyPI**: `twine upload dist/*`
- **Upload to TestPyPI**: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

## Project Structure
- **fred_forecaster/**: Main package code
  - **models/**: Forecasting models (SARIMAX, Bayesian)
  - **data.py**: FRED data retrieval
  - **calibration.py**: Simulation calibration to targets
  - **visualization.py**: Plotting functions using Plotly
- **demo/**: Streamlit demo application
- **tests/**: Unit tests

## Code Style Guidelines
- **Imports**: Standard libraries first, third-party packages second, local modules last
- **Types**: Use type annotations for all function parameters and return values
- **Docstrings**: Document all functions with descriptions, parameters and return values
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Error handling**: Include input validation and explicit error messages
- **Line length**: Maximum 88 characters (Black default)
- **Module organization**: Keep related functionality in dedicated modules
- **Formatting**: Run Black before committing changes
- **Testing**: Write unit tests for new functionality, mark slow tests with @pytest.mark.slow

## Common Pitfalls
- When working with pandas Period/Timestamp objects, use `pd.PeriodIndex` methods instead of arithmetic operations (e.g., avoid `last_period + 1`)
- For PyMC models, use the `dist()` API for initialization to avoid registration errors
- Use Plotly for visualizations rather than Matplotlib for interactive web compatibility
- SARIMAX models occasionally give convergence warnings but still produce useful forecasts
- When packaging for PyPI, use hyphens in the package name for PyPI but underscores for imports

## Project Overview
The fred-forecaster is a Python package for economic time series forecasting using Federal Reserve Economic Data (FRED). It provides:

1. Flexible data retrieval from FRED API
2. Multiple forecasting approaches:
   - Classical SARIMAX for traditional time series modeling
   - Bayesian structural time series for better uncertainty quantification
3. Forecast calibration to match external targets (e.g., CBO projections)
4. Interactive visualizations using Plotly
5. Streamlit demo application for easy exploration

The package is designed to be used as a library in other projects or via the included demo application.