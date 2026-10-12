import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def simplex_lmo(grad):
    """
    Solves the Linear Minimization Oracle (LMO) over the standard unit simplex.
    The minimizer on the simplex is the vertex corresponding to the smallest component of the gradient.
    
    Args:
        grad (numpy.array): Gradient vector of the objective function.
        
    Returns:
        tuple: A tuple containing:
            - s : The selected vertex (indicator vector).
            - i : The index of the selected asset.
    """
    n = len(grad)
    i = np.argmin(grad)
    s = np.zeros(n)
    s[i] = 1.0
    return s, i


def initial_point_simplex(n):
    """
    Generates the initial feasible point in the unit simplex (equally weighted portfolio).
    
    Args:
        n: Number of assets.
        
    Returns:
         Initial weight vector where each asset has weight 1/n.
    """
    return np.ones(n) / n


def portfolio_objective(x, mu, Sigma, gamma):
    """
    Computes the value of the portfolio optimization objective function (mean-variance model).
    
    Args:
        x : Vector of portfolio weights.
        mu: Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        
    Returns:
        the objective function value.
    """
    return gamma * (x @ Sigma @ x) - (mu @ x)


def portfolio_gradient(x, mu, Sigma, gamma):
    """
    Computes the gradient of the portfolio optimization objective function.
    
    Args:
        x : vector of portfolio weights.
        mu : vector of expected returns.
        Sigma : covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        
    Returns:
         the gradient vector.
    """
    return 2 * gamma * (Sigma @ x) - mu


def fw_gap(x, grad, s):
    """
    Computes the Frank-Wolfe duality gap.
    
    Args:
        x: Current point.
        grad: Gradient at the current point.
        s : Vertex selected by the LMO.
        
    Returns:
         The Frank-Wolfe gap.
    """
    return grad @ (x - s)


def exact_line_search_portfolio(x, d, grad, Sigma, gamma, alpha_max=1.0):
    """
    Performs exact line search for the quadratic portfolio objective function.
    
    Args:
        x : Current point.
        d : Search direction.
        grad : Gradient at the current point.
        Sigma  : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        alpha_max : Maximum feasible stepsize. Defaults to 1.0.
        
    Returns:
         the optimal stepsize.
    """
    denominator = 2 * gamma * (d @ Sigma @ d)

    # Handle cases where the denominator is zero or very close to zero
    if denominator <= 1e-15:
        if grad @ d < 0:
            return alpha_max
        else:
            return 0.0

    alpha = - (grad @ d) / denominator
    alpha = np.clip(alpha, 0.0, alpha_max)
    return alpha


def frank_wolfe_step(x, mu, Sigma, gamma):
    """
    Performs a single step of the classical Frank-Wolfe algorithm.
    
    Args:
        x : Current portfolio weights.
        mu : Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        
    Returns:
         A tuple containing:
            - x_new : Updated portfolio weights.
            - gap : Frank-Wolfe gap.
            - alpha  Chosen stepsize.
            - chosen_index : Index of the asset selected by the LMO.
    """
    grad = portfolio_gradient(x, mu, Sigma, gamma)
    s, chosen_index = simplex_lmo(grad)
    gap = fw_gap(x, grad, s)
    d = s - x
    alpha = exact_line_search_portfolio(
        x=x,
        d=d,
        grad=grad,
        Sigma=Sigma,
        gamma=gamma,
        alpha_max=1.0
    )
    x_new = x + alpha * d

    return x_new, gap, alpha, chosen_index


def frank_wolfe_portfolio(mu, Sigma, gamma, max_iter=1000, tol=1e-8):
    """
    Solves the portfolio optimization problem using the classical Frank-Wolfe method.
    
    Args:
        mu : Vector of expected returns.
        Sigma : covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        max_iter : Maximum number of iterations. Defaults to 1000.
        tol : Stopping tolerance for the Frank-Wolfe gap. Defaults to 1e-8.
        
    Returns:
       A tuple containing:
            - x : The approximate optimal portfolio weights.
            - history : A dictionary tracking the optimization progress (objective, gap, time, etc.).
    """
    n = len(mu)
    x = initial_point_simplex(n)

    history = {
        "objective": [],
        "gap": [],
        "alpha": [],
        "chosen_index": [],
        "risk": [],
        "return": [],
        "active_assets": [],
        "time": []
    }

    start_time = time.time()

    for k in range(max_iter):
        grad = portfolio_gradient(x, mu, Sigma, gamma)
        s, chosen_index = simplex_lmo(grad)
        d = s - x
        gap = fw_gap(x, grad, s)
        obj_value = portfolio_objective(x, mu, Sigma, gamma)
        portfolio_risk = x @ Sigma @ x
        portfolio_return = mu @ x
        active_assets = np.sum(x > 1e-12)
        elapsed_time = time.time() - start_time

        history["objective"].append(obj_value)
        history["gap"].append(gap)
        history["chosen_index"].append(chosen_index)
        history["risk"].append(portfolio_risk)
        history["return"].append(portfolio_return)
        history["active_assets"].append(active_assets)
        history["time"].append(elapsed_time)

        # Check stopping criterion
        if gap <= tol:
            break

        alpha = exact_line_search_portfolio(
            x=x,
            d=d,
            grad=grad,
            Sigma=Sigma,
            gamma=gamma,
            alpha_max=1.0
        )

        history["alpha"].append(alpha)

        x = x + alpha * d

    return x, history


def away_vertex_simplex(x, grad, tol_active=1e-12):
    """
    Identifies the away vertex (the worst active vertex) for the Away-Step Frank-Wolfe algorithm.
    
    Args:
        x : Current portfolio weights.
        grad : Gradient at the current point.
        tol_active : Tolerance for considering an asset active. Defaults to 1e-12.
        
    Returns:
         A tuple containing:
            - v : The selected away vertex (indicator vector).
            - j : The index of the selected away asset.
            - active_indices : Array of indices of currently active assets.
    """
    active_indices = np.where(x > tol_active)[0]
    # The away vertex maximizes the gradient among active assets
    j = active_indices[np.argmax(grad[active_indices])]

    v = np.zeros_like(x)
    v[j] = 1.0

    return v, j, active_indices


def away_step_frank_wolfe_step(x, mu, Sigma, gamma, tol_active=1e-12):
    """
    Performs a single step of the Away-Step Frank-Wolfe algorithm.
    
    Args:
        x : Current portfolio weights.
        mu : Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        tol_active : Tolerance for considering an asset active. Defaults to 1e-12.
        
    Returns:
         A tuple containing:
            - x_new : Updated portfolio weights.
            - gap : Frank-Wolfe gap.
            - alpha : Chosen stepsize.
            - fw_index : Index of the asset selected by the LMO (FW step).
            - away_index : Index of the away asset (away step).
            - direction_type : type of direction taken ("FW" or "Away").
    """
    grad = portfolio_gradient(x, mu, Sigma, gamma)
    s, fw_index = simplex_lmo(grad)
    d_fw = s - x
    gap = fw_gap(x, grad, s)
    v, away_index, active_indices = away_vertex_simplex(
        x=x,
        grad=grad,
        tol_active=tol_active
    )

    d_away = x - v

    # Evaluate the potential decrease for both directions
    fw_decrease = -grad @ d_fw
    away_decrease = -grad @ d_away

    # Choose the direction that offers the largest local decrease
    if fw_decrease >= away_decrease:
        d = d_fw
        direction_type = "FW"
        alpha_max = 1.0

    else:
        d = d_away
        direction_type = "Away"
        alpha_v = x[away_index]

        # Calculate maximum feasible stepsize for an away step
        if alpha_v >= 1.0 - tol_active:
            alpha_max = 0.0
        else:
            alpha_max = alpha_v / (1.0 - alpha_v)

    alpha = exact_line_search_portfolio(
        x=x,
        d=d,
        grad=grad,
        Sigma=Sigma,
        gamma=gamma,
        alpha_max=alpha_max
    )

    x_new = x + alpha * d
    # Numerical stability: ensure weights remain non-negative and sum to 1
    x_new = np.maximum(x_new, 0.0)
    x_new = x_new / np.sum(x_new)

    return x_new, gap, alpha, fw_index, away_index, direction_type


def away_step_frank_wolfe_portfolio(
    mu,
    Sigma,
    gamma,
    max_iter=1000,
    tol=1e-8,
    tol_active=1e-12
):
    """
    Solves the portfolio optimization problem using the Away-Step Frank-Wolfe method.
    
    Args:
        mu : Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma: Risk-aversion parameter.
        max_iter : Maximum number of iterations. Defaults to 1000.
        tol : Stopping tolerance for the Frank-Wolfe gap. Defaults to 1e-8.
        tol_active : Tolerance for considering an asset active. Defaults to 1e-12.
        
    Returns:
        A tuple containing:
            - x : The approximate optimal portfolio weights.
            - history : A dictionary tracking the optimization progress.
    """
    n = len(mu)
    x = initial_point_simplex(n)

    history = {
        "objective": [],
        "gap": [],
        "alpha": [],
        "fw_index": [],
        "away_index": [],
        "direction_type": [],
        "risk": [],
        "return": [],
        "active_assets": [],
        "time": []
    }

    start_time = time.time()

    for k in range(max_iter):
        obj_value = portfolio_objective(x, mu, Sigma, gamma)
        portfolio_risk = x @ Sigma @ x
        portfolio_return = mu @ x
        active_assets = np.sum(x > tol_active)
        elapsed_time = time.time() - start_time

        x_new, gap, alpha, fw_index, away_index, direction_type = away_step_frank_wolfe_step(
            x=x,
            mu=mu,
            Sigma=Sigma,
            gamma=gamma,
            tol_active=tol_active
        )

        history["objective"].append(obj_value)
        history["gap"].append(gap)
        history["alpha"].append(alpha)
        history["fw_index"].append(fw_index)
        history["away_index"].append(away_index)
        history["direction_type"].append(direction_type)
        history["risk"].append(portfolio_risk)
        history["return"].append(portfolio_return)
        history["active_assets"].append(active_assets)
        history["time"].append(elapsed_time)

        # Check stopping criterion
        if gap <= tol:
            break

        x = x_new

    return x, history


def pairwise_frank_wolfe_step(x, mu, Sigma, gamma, tol_active=1e-12):
    """
    Performs a single step of the Pairwise Frank-Wolfe algorithm.
    
    Args:
        x : Current portfolio weights.
        mu : Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        tol_active : Tolerance for considering an asset active. Defaults to 1e-12.
        
    Returns:
         A tuple containing:
            - x_new : Updated portfolio weights.
            - gap : Frank-Wolfe gap.
            - alpha : Chosen stepsize.
            - fw_index : Index of the asset selected by the LMO.
            - away_index : Index of the away asset.
    """
    grad = portfolio_gradient(x, mu, Sigma, gamma)
    s, fw_index = simplex_lmo(grad)
    v, away_index, active_indices = away_vertex_simplex(
        x=x,
        grad=grad,
        tol_active=tol_active
    )

    # The pairwise direction transfers weight from the away vertex to the FW vertex
    d = s - v
    gap = fw_gap(x, grad, s)
    # Maximum feasible stepsize is bounded by the weight of the away asset
    alpha_max = x[away_index]

    alpha = exact_line_search_portfolio(
        x=x,
        d=d,
        grad=grad,
        Sigma=Sigma,
        gamma=gamma,
        alpha_max=alpha_max
    )

    x_new = x + alpha * d

    # Numerical stability: ensure weights remain non-negative and sum to 1
    x_new = np.maximum(x_new, 0.0)
    x_new = x_new / np.sum(x_new)

    return x_new, gap, alpha, fw_index, away_index


def pairwise_frank_wolfe_portfolio(
    mu,
    Sigma,
    gamma,
    max_iter=1000,
    tol=1e-8,
    tol_active=1e-12
):
    """
    Solves the portfolio optimization problem using the Pairwise Frank-Wolfe method.
    
    Args:
        mu : Vector of expected returns.
        Sigma : Covariance matrix of asset returns.
        gamma : Risk-aversion parameter.
        max_iter : Maximum number of iterations. Defaults to 1000.
        tol: Stopping tolerance for the Frank-Wolfe gap. Defaults to 1e-8.
        tol_active : Tolerance for considering an asset active. Defaults to 1e-12.
        
    Returns:
         A tuple containing:
            - x (numpy.array): The approximate optimal portfolio weights.
            - history (dict): A dictionary tracking the optimization progress.
    """
    n = len(mu)
    x = initial_point_simplex(n)

    history = {
        "objective": [],
        "gap": [],
        "alpha": [],
        "fw_index": [],
        "away_index": [],
        "risk": [],
        "return": [],
        "active_assets": [],
        "time": []
    }

    start_time = time.time()

    for k in range(max_iter):
        obj_value = portfolio_objective(x, mu, Sigma, gamma)
        portfolio_risk = x @ Sigma @ x
        portfolio_return = mu @ x
        active_assets = np.sum(x > tol_active)
        elapsed_time = time.time() - start_time

        x_new, gap, alpha, fw_index, away_index = pairwise_frank_wolfe_step(
            x=x,
            mu=mu,
            Sigma=Sigma,
            gamma=gamma,
            tol_active=tol_active
        )

        history["objective"].append(obj_value)
        history["gap"].append(gap)
        history["alpha"].append(alpha)
        history["fw_index"].append(fw_index)
        history["away_index"].append(away_index)
        history["risk"].append(portfolio_risk)
        history["return"].append(portfolio_return)
        history["active_assets"].append(active_assets)
        history["time"].append(elapsed_time)

        # Check stopping criterion
        if gap <= tol:
            break

        x = x_new

    return x, history


def portfolio_metrics(x, mu, Sigma, gamma, tol_active=1e-12):
    """
    Computes summary metrics for a given portfolio allocation.
    
    Args:
        x : Portfolio weights.
        mu : Expected returns.
        Sigma : Covariance matrix.
        gamma : Risk-aversion parameter.
        tol_active : Tolerance for active assets ( 1e-12 ).
        
    Returns:
        A dictionary containing portfolio metrics (objective, risk, return, active assets, etc.).
    """
    objective_value = portfolio_objective(x, mu, Sigma, gamma)
    portfolio_risk = x @ Sigma @ x
    portfolio_return = mu @ x
    active_assets = np.sum(x > tol_active)

    metrics = {
        "objective": objective_value,
        "risk": portfolio_risk,
        "return": portfolio_return,
        "active_assets": active_assets,
        "sum_weights": np.sum(x),
        "min_weight": np.min(x)
    }

    return metrics


def run_all_methods_portfolio(
    mu,
    Sigma,
    gamma,
    max_iter=1000,
    tol=1e-8,
    tol_active=1e-12
):
    """
    Runs and compares all three Frank-Wolfe variants on the portfolio optimization problem.
    
    Args:
        mu : Expected returns.
        Sigma : Covariance matrix.
        gamma : Risk-aversion parameter.
        max_iter : Maximum number of iterations. Defaults to 1000.
        tol : Stopping tolerance. Defaults to 1e-8.
        tol_active : Tolerance for active assets ( 1e-12 ).
        
    Returns:
        A dictionary containing the results (final weights, history, metrics) for each method.
    """
    results = {}

    # 1. Classical Frank-Wolfe
    x_fw, history_fw = frank_wolfe_portfolio(
        mu=mu,
        Sigma=Sigma,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol
    )

    results["FW"] = {
        "x": x_fw,
        "history": history_fw,
        "metrics": portfolio_metrics(
            x=x_fw,
            mu=mu,
            Sigma=Sigma,
            gamma=gamma,
            tol_active=tol_active
        )
    }

    # 2. Away-Step Frank-Wolfe
    x_afw, history_afw = away_step_frank_wolfe_portfolio(
        mu=mu,
        Sigma=Sigma,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        tol_active=tol_active
    )

    results["Away-Step FW"] = {
        "x": x_afw,
        "history": history_afw,
        "metrics": portfolio_metrics(
            x=x_afw,
            mu=mu,
            Sigma=Sigma,
            gamma=gamma,
            tol_active=tol_active
        )
    }

    # 3. Pairwise Frank-Wolfe
    x_pfw, history_pfw = pairwise_frank_wolfe_portfolio(
        mu=mu,
        Sigma=Sigma,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        tol_active=tol_active
    )

    results["Pairwise FW"] = {
        "x": x_pfw,
        "history": history_pfw,
        "metrics": portfolio_metrics(
            x=x_pfw,
            mu=mu,
            Sigma=Sigma,
            gamma=gamma,
            tol_active=tol_active
        )
    }

    return results


def results_summary_table(results):
    """
    Creates a pandas DataFrame summarizing the results of the different methods.
    
    Args:
        results : dictionary of results generated by run_all_methods_portfolio.
        
    Returns:
        Summary table of the metrics.
    """
    rows = []

    for method_name, result in results.items():
        history = result["history"]
        metrics = result["metrics"]
        n_iter = len(history["objective"])

        if n_iter > 0:
            final_gap = history["gap"][-1]
            total_time = history["time"][-1]
        else:
            final_gap = np.nan
            total_time = np.nan

        row = {
            "method": method_name,
            "objective": metrics["objective"],
            "risk": metrics["risk"],
            "return": metrics["return"],
            "active_assets": metrics["active_assets"],
            "iterations": n_iter,
            "final_gap": final_gap,
            "time": total_time,
            "sum_weights": metrics["sum_weights"],
            "min_weight": metrics["min_weight"]
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


# =============================================================================
# DATASET 1: FF49Industries
# =============================================================================
file_path = "FF49Industries.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data = pd.read_excel(file_path, header=None)
data.columns = [f"Asset_{i+1}" for i in range(data.shape[1])]
returns = data.copy()
returns = returns.apply(pd.to_numeric, errors="coerce")
returns = returns.dropna(axis=0)
returns = returns.dropna(axis=1)

# Compute expected returns and covariance matrix
mu = returns.mean().values
Sigma = returns.cov().values
Sigma = 0.5 * (Sigma + Sigma.T)  # Ensure symmetry

print("Numero osservazioni:", returns.shape[0])
print("Numero asset:", returns.shape[1])
print("Dimensione mu:", mu.shape)
print("Dimensione Sigma:", Sigma.shape)

gamma = 4.0

results_ff49 = run_all_methods_portfolio(
    mu=mu,
    Sigma=Sigma,
    gamma=gamma,
    max_iter=1000,
    tol=1e-8
)

summary = results_summary_table(results_ff49)
summary
print(summary)

def final_weights_table(results, asset_names=None, tol_active=1e-12):
    """
    Creates a DataFrame of the final active asset weights for each method.
    """
    rows = []
    for method_name, result in results.items():
        x = result["x"]
        for i, weight in enumerate(x):
            if weight > tol_active:
                asset = asset_names[i] if asset_names is not None else f"Asset_{i+1}"
                rows.append({
                    "method": method_name,
                    "asset": asset,
                    "weight": weight
                })
    return pd.DataFrame(rows)

asset_names = list(returns.columns)
weights_table = final_weights_table(
    results=results_ff49,
    asset_names=asset_names
)
weights_table
print(weights_table)


# =============================================================================
# DATASET 2: NASDAQ100
# =============================================================================
file_path = "NASDAQ100.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data_nasdaq = pd.read_excel(file_path, header=None)
data_nasdaq.columns = [f"Asset_{i+1}" for i in range(data_nasdaq.shape[1])]
returns_nasdaq = data_nasdaq.copy()
returns_nasdaq = returns_nasdaq.apply(pd.to_numeric, errors="coerce")
returns_nasdaq = returns_nasdaq.dropna(axis=0)
returns_nasdaq = returns_nasdaq.dropna(axis=1)

# Compute expected returns and covariance matrix
mu_nasdaq = returns_nasdaq.mean().values
Sigma_nasdaq = returns_nasdaq.cov().values
Sigma_nasdaq = 0.5 * (Sigma_nasdaq + Sigma_nasdaq.T) # Ensure symmetry

print("Dataset: NASDAQ100")
print("Numero osservazioni:", returns_nasdaq.shape[0])
print("Numero asset:", returns_nasdaq.shape[1])
print("Dimensione mu:", mu_nasdaq.shape)
print("Dimensione Sigma:", Sigma_nasdaq.shape)

results_nasdaq = run_all_methods_portfolio(
    mu=mu_nasdaq,
    Sigma=Sigma_nasdaq,
    gamma=gamma,
    max_iter=1000,
    tol=1e-8
)

summary_nasdaq = results_summary_table(results_nasdaq)
summary_nasdaq
print(summary_nasdaq)

asset_names_nasdaq = list(returns_nasdaq.columns)
weights_table_nasdaq = final_weights_table(
    results=results_nasdaq,
    asset_names=asset_names_nasdaq
)
weights_table_nasdaq
print(weights_table_nasdaq)


# =============================================================================
# DATASET 3: FTSE100
# =============================================================================
file_path = "FTSE100.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data_ftse = pd.read_excel(file_path, header=None)
data_ftse.columns = [f"Asset_{i+1}" for i in range(data_ftse.shape[1])]
returns_ftse = data_ftse.copy()
returns_ftse = returns_ftse.apply(pd.to_numeric, errors="coerce")
returns_ftse = returns_ftse.dropna(axis=0)
returns_ftse = returns_ftse.dropna(axis=1)

# Compute expected returns and covariance matrix
mu_ftse = returns_ftse.mean().values
Sigma_ftse = returns_ftse.cov().values
Sigma_ftse = 0.5 * (Sigma_ftse + Sigma_ftse.T) # Ensure symmetry

print("Dataset: FTSE100")
print("Numero osservazioni:", returns_ftse.shape[0])
print("Numero asset:", returns_ftse.shape[1])
print("Dimensione mu:", mu_ftse.shape)
print("Dimensione Sigma:", Sigma_ftse.shape)

results_ftse = run_all_methods_portfolio(
    mu=mu_ftse,
    Sigma=Sigma_ftse,
    gamma=gamma,
    max_iter=1000,
    tol=1e-8
)

summary_ftse = results_summary_table(results_ftse)
summary_ftse
print(summary_ftse)

asset_names_ftse = list(returns_ftse.columns)
weights_table_ftse = final_weights_table(
    results=results_ftse,
    asset_names=asset_names_ftse
)
weights_table_ftse
print(weights_table_ftse)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_objective_vs_iterations_clear(results, save_path=None):
    """Plots the objective function value across iterations."""
    plt.figure(figsize=(10, 6))
    plot_style = {
        "FW": {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 3, "offset": -0.06, "zorder": 4},
        "Away-Step FW": {"marker": "s", "linestyle": "-", "linewidth": 2, "markersize": 3, "offset": 0.06, "zorder": 3},
        "Pairwise FW": {"marker": "^", "linestyle": "-", "linewidth": 2, "markersize": 5, "offset": 0.0, "zorder": 2}
    }

    for method_name, result in results.items():
        objective_values = np.array(result["history"]["objective"])
        iterations = np.arange(len(objective_values))
        style = plot_style.get(method_name, {"marker": "o", "linestyle": "-", "linewidth": 1.5, "markersize": 5, "offset": 0.0, "zorder": 1})
        iterations_plot = iterations + style["offset"]

        plt.plot(
            iterations_plot, objective_values, marker=style["marker"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], markersize=style["markersize"],
            label=f"{method_name} ({len(objective_values)} iter.)", zorder=style["zorder"]
        )

    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel(r"Objective value $f(x_k)$", fontsize=13)
    plt.title("Objective value vs iterations", fontsize=15, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_objective_vs_time(results, save_path=None):
    """Plots the objective function value against computational time."""
    plt.figure(figsize=(10, 6))
    plot_style = {
        "FW": {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 3, "offset": -0.06, "zorder": 4},
        "Away-Step FW": {"marker": "s", "linestyle": "-", "linewidth": 2, "markersize": 3, "offset": 0.06, "zorder": 3},
        "Pairwise FW": {"marker": "^", "linestyle": "-", "linewidth": 2, "markersize": 5, "offset": 0.0, "zorder": 2}
    }

    for method_name, result in results.items():
        objective_values = np.array(result["history"]["objective"])
        time_values = np.array(result["history"]["time"])
        style = plot_style.get(method_name, {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 5, "zorder": 1})

        plt.plot(
            time_values, objective_values, marker=style["marker"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], markersize=style["markersize"],
            label=f"{method_name} ({len(objective_values)} iter.)", zorder=style["zorder"]
        )

    plt.xlabel("Time (seconds)", fontsize=13)
    plt.ylabel(r"Objective value $f(x_k)$", fontsize=13)
    plt.title("Objective value vs computational time", fontsize=15, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_iterations_comparison_all(all_results_dict, save_path=None):
    """Creates a bar chart comparing the number of iterations across all datasets."""
    methods = ["FW", "Away-Step FW", "Pairwise FW"]
    colors = {'FF49': 'skyblue', 'Nasdaq100': 'red', 'FTSE100': 'green'}
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (ds_name, results) in enumerate(all_results_dict.items()):
        iters = [len(results[m]["history"]["objective"]) for m in methods]
        pos = x + (i - 1) * width
        color = colors.get(ds_name, None)
        bars = ax.bar(pos, iters, width, label=ds_name, color=color, edgecolor='black', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Algorithm', fontsize=13)
    ax.set_ylabel('Number of iterations', fontsize=13)
    ax.set_title('Iterations comparison across datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(title="Datasets", fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_fw_gap_vs_iterations_clear(results, save_path=None):
    """Plots the Frank-Wolfe duality gap (log scale) across iterations."""
    plt.figure(figsize=(10, 6))
    plot_style = {
        "FW": {"marker": "o", "linestyle": "-", "linewidth": 1, "markersize": 1.5, "offset": -0.06, "zorder": 4},
        "Away-Step FW": {"marker": "s", "linestyle": "-", "linewidth": 2, "markersize": 3, "offset": 0.06, "zorder": 3},
        "Pairwise FW": {"marker": "^", "linestyle": "-", "linewidth": 2, "markersize": 5, "offset": 0.0, "zorder": 2}
    }

    for method_name, result in results.items():
        gap_values = np.array(result["history"]["gap"])
        gap_values = np.maximum(gap_values, 1e-16) # Avoid log(0)
        iterations = np.arange(len(gap_values))
        style = plot_style.get(method_name, {"marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 5, "offset": 0.0, "zorder": 1})
        iterations_plot = iterations + style["offset"]

        plt.semilogy(
            iterations_plot, gap_values, marker=style["marker"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], markersize=style["markersize"],
            label=f"{method_name} ({len(gap_values)} iter.)", zorder=style["zorder"]
        )

    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel(r"Frank-Wolfe gap $G(x_k)$", fontsize=13)
    plt.title("Frank-Wolfe gap vs iterations", fontsize=15, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_active_assets_comparison(all_results_dict, save_path=None):
    """Creates a bar chart comparing the final number of active assets across datasets."""
    methods = ["FW", "Away-Step FW", "Pairwise FW"]
    colors = ['blue', 'gold', 'gray']
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (ds_name, results) in enumerate(all_results_dict.items()):
        active = [results[m]["metrics"]["active_assets"] for m in methods]
        pos = x + (i - 1) * width
        ax.bar(pos, active, width, label=ds_name, color=colors[i % len(colors)], edgecolor='black', alpha=0.8)

        for j, val in enumerate(active):
            ax.annotate(f'{int(val)}', xy=(pos[j], val), xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Algorithm', fontsize=13)
    ax.set_ylabel('Number of Active Assets', fontsize=13)
    ax.set_title('Active assets comparison across datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(title="Datasets", fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Generate and save plots for each dataset
plot_objective_vs_iterations_clear(results_ff49, save_path="objective_iteration_ff49")
plot_objective_vs_time(results_ff49, save_path="objective_time_ff49")
plot_fw_gap_vs_iterations_clear(results_ff49, save_path="fwgap_iterations_ff49")

plot_objective_vs_iterations_clear(results_nasdaq, save_path="objective_iteration_nasdaq")
plot_objective_vs_time(results_nasdaq, save_path="objective_time_nasdaq")
plot_fw_gap_vs_iterations_clear(results_nasdaq, save_path="fwgap_iterations_nasdaq")

plot_objective_vs_iterations_clear(results_ftse, save_path="objective_iteration_ftse")
plot_objective_vs_time(results_ftse, save_path="objective_time_ftse")
plot_fw_gap_vs_iterations_clear(results_ftse, save_path="fwgap_iterations_ftse")

# Generate and save comparison plots
all_results = {
    'FF49': results_ff49,
    'Nasdaq100': results_nasdaq,
    'FTSE100': results_ftse
}
plot_iterations_comparison_all(all_results, save_path="iterations_comparison_all_datasets.png")
plot_active_assets_comparison(all_results, save_path="active_assets_comparison.png")