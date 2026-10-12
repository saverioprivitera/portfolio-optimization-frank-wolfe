# Portfolio Optimization: Projection-Free Optimization Methods

This repository explores the application of projection-free optimization algorithms to solve the portfolio selection problem based on the classical mean-variance model. 

## The Problem
The goal of portfolio optimization is to determine the best allocation of capital among a set of financial assets to achieve an optimal balance between expected return and risk. 

We minimize a quadratic, risk-adjusted penalty function:
$$\min_{x \in \Delta_{n-1}} f(x) = \gamma x^T \Sigma x - \mu^T x$$

Where:
*   $x$ is the vector of portfolio weights.
*   $\mu$ is the vector of historical mean returns.
*   $\Sigma$ is the covariance matrix of asset returns.
*   $\gamma$ is the risk-aversion parameter (set to 4.0 in this project).
*   $\Delta_{n-1}$ is the standard unit simplex, ensuring that the weights are non-negative (no short-selling) and sum to 1 (full investment).

## Algorithms Implemented
The specific geometry of the unit simplex makes the Frank-Wolfe (FW) family of algorithms highly suitable, as they rely on a simple Linear Minimization Oracle (LMO) instead of expensive Euclidean projections.

This repository contains Python implementations for:
1.  **Classical Frank-Wolfe (FW):** Moves towards the vertex that locally gives the largest decrease of the objective function. It often suffers from "zig-zagging" and slow convergence.
2.  **Away-Step Frank-Wolfe (AFW):** Extends the classical method by allowing the algorithm to remove weight from less convenient *active* assets, moving "away" from bad vertices.
3.  **Pairwise Frank-Wolfe (PFW):** Transfers weight directly from a bad active vertex (away vertex) to a promising vertex (FW vertex), proving to be highly efficient.

All methods utilize exact line search to determine the optimal step size at each iteration.

## Repository Contents
*   `portfolio_o.py`: The main Python script containing the implementation of the objective functions, gradients, LMO, line search algorithms, and the three Frank-Wolfe variants. It also includes functions to run experiments and plot the results.

## Experiments and Key Results
The algorithms were evaluated on three real-world datasets: FF49, NASDAQ100, and FTSE100. The comparison focused on convergence speed, the final Frank-Wolfe gap, and portfolio sparsity (number of active assets).

**Summary of Findings:**
*   **Classical FW** often hits the iteration limit (1000 iterations) without reaching the required stopping tolerance ($10^{-8}$), resulting in non-sparse portfolios with many active assets.
*   **Active-set variants (AFW and PFW)** significantly outperform the classical method. They converge much faster (e.g., ~70-150 iterations), achieve a much smaller optimality gap, and produce highly sparse portfolios (e.g., 8-14 active assets instead of 80+).
*   **Pairwise Frank-Wolfe** it also produces highly sparse portfolios and it is the fastest variant in terms of both iteration count and CPU time across the tested datasets.
