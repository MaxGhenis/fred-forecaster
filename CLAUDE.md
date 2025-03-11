# Fred Forecaster Development Guide

## Commands
- **Run app**: `streamlit run app.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Environment setup**: Set `FRED_API_KEY` environment variable
- **Format code**: `black .`
- **Check types**: `mypy src/ app.py`
- **Lint code**: `flake8 src/ app.py`

## Code Style Guidelines
- **Imports**: Standard libraries first, third-party packages second, local modules last
- **Types**: Use type annotations for all function parameters and return values
- **Docstrings**: Document all functions with descriptions, parameters and return values
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Error handling**: Include input validation and explicit error messages
- **Line length**: Maximum 88 characters (Black default)
- **Module organization**: Keep related functionality in dedicated modules
- **Formatting**: Run Black before committing changes

The codebase is structured as a Streamlit application that fetches FRED data, generates time-series forecasts with SARIMAX models, and visualizes forecast distributions.