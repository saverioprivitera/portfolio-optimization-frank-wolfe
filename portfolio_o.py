import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time



def simplex_lmo(grad):
    n = len(grad)
    i = np.argmin(grad)
    s = np.zeros(n)
    s[i] = 1.0
    return s, i



def initial_point_simplex(n):
    return np.ones(n) / n



def portfolio_objective(x, mu, Sigma, gamma):
    return gamma * (x @ Sigma @ x) - (mu @ x)



def portfolio_gradient(x, mu, Sigma, gamma):
    return 2 * gamma * (Sigma @ x) - mu



def fw_gap(x, grad, s):
    return grad @ (x - s)



def exact_line_search_portfolio(x, d, grad, Sigma, gamma, alpha_max=1.0):
    denominator = 2 * gamma * (d @ Sigma @ d)

    if denominator <= 1e-15:
        if grad @ d < 0:
            return alpha_max
        else:
            return 0.0

    alpha = - (grad @ d) / denominator
    alpha = np.clip(alpha, 0.0, alpha_max)
    return alpha




def frank_wolfe_step(x, mu, Sigma, gamma):
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

    active_indices = np.where(x > tol_active)[0]
    j = active_indices[np.argmax(grad[active_indices])]

    v = np.zeros_like(x)
    v[j] = 1.0

    return v, j, active_indices





def away_step_frank_wolfe_step(x, mu, Sigma, gamma, tol_active=1e-12):
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

    fw_decrease = -grad @ d_fw
    away_decrease = -grad @ d_away

    if fw_decrease >= away_decrease:
        d = d_fw
        direction_type = "FW"
        alpha_max = 1.0

    else:
        d = d_away
        direction_type = "Away"
        alpha_v = x[away_index]

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

        if gap <= tol:
            break

        x = x_new

    return x, history





def pairwise_frank_wolfe_step(x, mu, Sigma, gamma, tol_active=1e-12):

    grad = portfolio_gradient(x, mu, Sigma, gamma)
    s, fw_index = simplex_lmo(grad)
    v, away_index, active_indices = away_vertex_simplex(
        x=x,
        grad=grad,
        tol_active=tol_active
    )

    d = s - v
    gap = fw_gap(x, grad, s)
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

        if gap <= tol:
            break

        x = x_new

    return x, history





def portfolio_metrics(x, mu, Sigma, gamma, tol_active=1e-12):

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

    results = {}

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






file_path = "FF49Industries.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data = pd.read_excel(file_path, header=None)

data.columns = [f"Asset_{i+1}" for i in range(data.shape[1])]
returns = data.copy()
returns = returns.apply(pd.to_numeric, errors="coerce")
returns = returns.dropna(axis=0)
returns = returns.dropna(axis=1)
mu = returns.mean().values
Sigma = returns.cov().values
Sigma = 0.5 * (Sigma + Sigma.T)

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





file_path = "NASDAQ100.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data_nasdaq = pd.read_excel(file_path, header=None)

data_nasdaq.columns = [f"Asset_{i+1}" for i in range(data_nasdaq.shape[1])]
returns_nasdaq = data_nasdaq.copy()
returns_nasdaq = returns_nasdaq.apply(pd.to_numeric, errors="coerce")
returns_nasdaq = returns_nasdaq.dropna(axis=0)
returns_nasdaq = returns_nasdaq.dropna(axis=1)
mu_nasdaq = returns_nasdaq.mean().values
Sigma_nasdaq = returns_nasdaq.cov().values
Sigma_nasdaq = 0.5 * (Sigma_nasdaq + Sigma_nasdaq.T)

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







file_path = "FTSE100.xlsx"
excel_file = pd.ExcelFile(file_path)
print(excel_file.sheet_names)

data_ftse = pd.read_excel(file_path, header=None)

data_ftse.columns = [f"Asset_{i+1}" for i in range(data_ftse.shape[1])]
returns_ftse = data_ftse.copy()
returns_ftse = returns_ftse.apply(pd.to_numeric, errors="coerce")
returns_ftse = returns_ftse.dropna(axis=0)
returns_ftse = returns_ftse.dropna(axis=1)
mu_ftse = returns_ftse.mean().values
Sigma_ftse = returns_ftse.cov().values
Sigma_ftse = 0.5 * (Sigma_ftse + Sigma_ftse.T)

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






def plot_objective_vs_iterations_clear(results, save_path=None):
    plt.figure(figsize=(10, 6))

    plot_style = {
        "FW": {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 3,
            "offset": -0.06,
            "zorder": 4
        },
        "Away-Step FW": {
            "marker": "s",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 3,
            "offset": 0.06,
            "zorder": 3
        },
        "Pairwise FW": {
            "marker": "^",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 5,
            "offset": 0.0,
            "zorder": 2
        }
    }

    for method_name, result in results.items():
        objective_values = np.array(result["history"]["objective"])
        iterations = np.arange(len(objective_values))

        style = plot_style.get(method_name, {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 1.5,
            "markersize": 5,
            "offset": 0.0,
            "zorder": 1
        })

        iterations_plot = iterations + style["offset"]

        plt.plot(
            iterations_plot,
            objective_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=f"{method_name} ({len(objective_values)} iter.)",
            zorder=style["zorder"]
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
    plt.figure(figsize=(10, 6))

    plot_style = {
        "FW": {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 3,
            "offset": -0.06,
            "zorder": 4
        },
        "Away-Step FW": {
            "marker": "s",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 3,
            "offset": 0.06,
            "zorder": 3
        },
        "Pairwise FW": {
            "marker": "^",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 5,
            "offset": 0.0,
            "zorder": 2
        }
    }

    for method_name, result in results.items():

        objective_values = np.array(result["history"]["objective"])
        time_values = np.array(result["history"]["time"])

        style = plot_style.get(method_name, {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 5,
            "zorder": 1
        })

        plt.plot(
            time_values,
            objective_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=f"{method_name} ({len(objective_values)} iter.)",
            zorder=style["zorder"]
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
    methods = ["FW", "Away-Step FW", "Pairwise FW"]
    datasets = list(all_results_dict.keys())
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
    plt.figure(figsize=(10, 6))

    plot_style = {
        "FW": {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 1,
            "markersize": 1.5,
            "offset": -0.06,
            "zorder": 4
        },
        "Away-Step FW": {
            "marker": "s",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 3,
            "offset": 0.06,
            "zorder": 3
        },
        "Pairwise FW": {
            "marker": "^",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 5,
            "offset": 0.0,
            "zorder": 2
        }
    }

    for method_name, result in results.items():

        gap_values = np.array(result["history"]["gap"])
        gap_values = np.maximum(gap_values, 1e-16)

        iterations = np.arange(len(gap_values))

        style = plot_style.get(method_name, {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 5,
            "offset": 0.0,
            "zorder": 1
        })

        iterations_plot = iterations + style["offset"]

        plt.semilogy(
            iterations_plot,
            gap_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=f"{method_name} ({len(gap_values)} iter.)",
            zorder=style["zorder"]
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
    methods = ["FW", "Away-Step FW", "Pairwise FW"]
    datasets = list(all_results_dict.keys())
    colors = ['blue', 'gold', 'gray']

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (ds_name, results) in enumerate(all_results_dict.items()):
        active = [results[m]["metrics"]["active_assets"] for m in methods]
        pos = x + (i - 1) * width
        ax.bar(pos, active, width, label=ds_name, color=colors[i % len(colors)], edgecolor='black', alpha=0.8)

        for j, val in enumerate(active):
            ax.annotate(f'{int(val)}',
                        xy=(pos[j], val),
                        xytext=(0, 3),
                        textcoords="offset points",
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




plot_objective_vs_iterations_clear(results_ff49, save_path="objective_iteration_ff49")
plot_objective_vs_time(results_ff49, save_path="objective_time_ff49")
plot_fw_gap_vs_iterations_clear(results_ff49, save_path="fwgap_iterations_ff49")



plot_objective_vs_iterations_clear(results_nasdaq, save_path="objective_iteration_nasdaq")
plot_objective_vs_time(results_nasdaq, save_path="objective_time_nasdaq")
plot_fw_gap_vs_iterations_clear(results_nasdaq, save_path="fwgap_iterations_nasdaq")



plot_objective_vs_iterations_clear(results_ftse, save_path="objective_iteration_ftse")
plot_objective_vs_time(results_ftse, save_path="objective_time_ftse")
plot_fw_gap_vs_iterations_clear(results_ftse, save_path="fwgap_iterations_ftse")




all_results = {
    'FF49': results_ff49,
    'Nasdaq100': results_nasdaq,
    'FTSE100': results_ftse
}

plot_iterations_comparison_all(all_results, save_path="iterations_comparison_all_datasets.png")
plot_active_assets_comparison(all_results, save_path="active_assets_comparison.png")