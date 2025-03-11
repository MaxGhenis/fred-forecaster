# Fred Forecaster Development Guide

## Commands
- **Run app**: `streamlit run app.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Environment setup**: Set `FRED_API_KEY` environment variable
- **Format code**: `black .`
- **Check types**: `mypy src/ app.py`
- **Lint code**: `flake8 src/ app.py`
- **Run tests**: `pytest tests/`
- **Run specific test**: `pytest tests/test_data.py::TestData::test_fetch_fred_data`
- **Run tests with coverage**: `pytest --cov=src tests/`
- **Run slow tests**: `pytest --run-slow tests/`

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

## Project Overview
The fred-forecaster is a Streamlit application that fetches Federal Reserve Economic Data (FRED), 
generates time-series forecasts using both classical (SARIMAX) and Bayesian models, 
and visualizes forecast distributions with optional calibration to CBO targets.