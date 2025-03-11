import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(df_quarterly, sim_array, forecast_index, weights=None):
    """
    Plot historical data + simulation paths + (optional) weighted means.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical
    ax.plot(
        df_quarterly.index.to_timestamp(),
        df_quarterly["Debt"],
        label="Historical",
        color="black",
    )

    # Convert forecast_index to timestamps
    time = forecast_index.to_timestamp()

    # Plot each simulation in the background (light grey)
    for j in range(sim_array.shape[1]):
        ax.plot(time, sim_array[:, j], color="grey", alpha=0.03)

    # Weighted mean or unweighted average
    if weights is not None:
        weighted_mean = np.dot(sim_array, weights)
        ax.plot(time, weighted_mean, color="blue", linewidth=2, label="Weighted Mean")

        # 5th-95th percentile
        lb = np.percentile(sim_array, 5, axis=1)
        ub = np.percentile(sim_array, 95, axis=1)
        ax.fill_between(time, lb, ub, color="blue", alpha=0.2, label="5-95% range")
    else:
        # Just do a simple percentile fill if no weights
        lb = np.percentile(sim_array, 5, axis=1)
        ub = np.percentile(sim_array, 95, axis=1)
        ax.fill_between(time, lb, ub, color="orange", alpha=0.2, label="5-95% range")

    ax.set_title("Debt Forecast Simulations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Debt (Trillions)")
    ax.grid(True)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_drop_probabilities(sim_array, forecast_index, weights=None):
    """
    Plot bar chart for quarter-over-quarter drop probabilities from 2025Q1 onward.
    Returns a matplotlib Figure.
    """
    # If no weights, assume uniform
    N = sim_array.shape[1]
    if weights is None:
        weights = np.ones(N) / N

    # Probability for each quarter from 2025Q1 onward
    prob_fall_data = []
    for i in range(1, len(forecast_index)):
        if forecast_index[i].year < 2025:
            continue
        arr_this = sim_array[i, :]
        arr_prev = sim_array[i - 1, :]
        prob = np.dot(weights, (arr_this < arr_prev).astype(float))
        prob_fall_data.append((forecast_index[i], prob))

    df_prob_fall = pd.DataFrame(
        prob_fall_data, columns=["Quarter", "ProbDebtFalls"]
    ).set_index("Quarter")

    # Overall probability of at least one drop
    diffs = np.diff(sim_array, axis=0)
    has_decline = (diffs < 0).any(axis=0).astype(float)
    overall_prob_drop = np.dot(weights, has_decline)

    # Probability from 2025Q1 onward (similarly)
    start_idx = None
    for i, q in enumerate(forecast_index):
        if q.year >= 2025:
            start_idx = i
            break
    relevant_diffs = np.diff(sim_array[start_idx:], axis=0)
    has_decline_2025 = (relevant_diffs < 0).any(axis=0).astype(float)
    prob_drop_2025on = np.dot(weights, has_decline_2025)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        df_prob_fall.index.astype(str), df_prob_fall["ProbDebtFalls"], color="orange"
    )
    ax.set_title("Quarter-over-Quarter Fall Probabilities (2025Q1+)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Probability")
    ax.set_xticklabels(df_prob_fall.index.astype(str), rotation=45)
    ax.text(
        0.01,
        0.9,
        f"Overall: {overall_prob_drop:.2%}\n2025+ window: {prob_drop_2025on:.2%}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.grid(axis="y")
    fig.tight_layout()
    return fig
