import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calibrate_simulations(
    sim_array: np.ndarray, forecast_index: pd.PeriodIndex
) -> np.ndarray:
    """
    Example: reweight sim_array so that Q4 of each year matches certain targets.
    Hard-coded CBO targets for demonstration.

    Returns a weight vector of length = number of simulations (N).
    """
    # Hard-coded CBO annual forecasts (in trillions)
    cbo_annual = {2024: 35.230, 2025: 37.209, 2026: 39.130, 2027: 40.872, 2028: 42.748}

    df_fc = pd.DataFrame(sim_array, index=forecast_index)
    # Filter to Q4 only.
    df_Q4 = df_fc[df_fc.index.quarter == 4]
    S = df_Q4.to_numpy()  # shape: (num_years, N)
    calib_years = df_Q4.index.year.to_numpy()

    def ssq_obj(w, S, years, targets):
        valid_indices = [i for i, y in enumerate(years) if y in targets]
        if not valid_indices:
            return np.inf
        T = np.array([targets[years[i]] for i in valid_indices])
        weighted_q4 = S[valid_indices, :].dot(w)
        return np.sum((weighted_q4 - T) ** 2)

    N_sims = sim_array.shape[1]
    w0 = np.ones(N_sims) / N_sims
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, None)] * N_sims

    res = minimize(
        fun=lambda w: ssq_obj(w, S, calib_years, cbo_annual),
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Calibration failed: {res.message}")

    weights = res.x
    return weights
