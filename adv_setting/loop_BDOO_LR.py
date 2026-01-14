r"""BDOO experiments for linear regression on the bodyfat dataset.

The script follows the implementation details provided in the paper notes:

* Dataset: bodyfat (14 features, 252 instances) loaded from
  ``data/adv_setting/bodyfat`` in LIBSVM format.
* Preprocessing: each feature is scaled to [-1, 1] with ``MaxAbsScaler`` and
  each sample is normalized to unit `\ell_2` length.
* Task: distributed online (regularized) linear regression with squared loss
  and optional L2 regularization over an `\ell_2` ball of radius 10.
* Network: default 8-node cycle graph (each node has two neighbors) with
  optional 9-node 3×3 grid and 8-node cube topologies.
* Time horizon: by default 20,000 rounds.

This version additionally computes:
  - Average loss per round (averaged over learners and normalized by T)
  - Average regret per round (normalized by T), where the comparator is the
    best fixed x in ||x||_2 <= R minimizing the same cumulative loss.

Loss definition (per learner i, round t):
    f_t^i(x) = (a_{i,t}^T x - b_{i,t})^2 + (alpha/2) ||x||^2
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from optimization_utils.BanditLogisticRegression import (
    _build_network_topology,
    _compute_bounds,
    _load_bodyfat_dataset,
    _make_loss_oracle,
    run_algorithm2_bandit_paper_params,
)
from optimization_utils.adv_setting_utils import compute_avg_loss_and_avg_regret_per_round


# Regret/comparator helpers are imported from optimization_utils.adv_setting_utils.


# ============================================================
# xi / radii selection (unchanged)
# ============================================================

def _compute_xi(
    *,
    setting: str,
    T: int,
    dim: int,
    num_nodes: int,
    R: float,
    r: float,
    L_f: float,
    C: float,
    rho: float,
    alpha: float | None,
) -> float:
    """Compute ξ = δ / r for the provided hyperparameters.

    This mirrors the formulas in ``run_algorithm2_bandit_paper_params`` to
    validate whether a given pair ``(R, r)`` is admissible.
    """
    if setting == "convex":
        c1 = 3.0 * dim * R * C * (1.0 + 4.0 * rho * (1.0 + np.sqrt(num_nodes)) / (1.0 - rho))
        c2 = 2.0 * (L_f + C / r)
        print("c1: ", c1)
        print("c2: ", c2)
        delta = np.sqrt(c1 / c2) * (T ** (-0.25))
    elif setting == "strongly_convex":
        if alpha is None or alpha <= 0:
            raise ValueError("alpha must be positive for strongly convex settings")
        c3 = (dim * (C**2) / (2.0 * alpha)) * (
            1.0 + 6.0 * rho * (1.0 + np.sqrt(num_nodes)) / (1.0 - rho)
        )
        c2 = 2.0 * (L_f + C / r)
        print("c3: ", c3)
        print("c2: ", c2)
        delta = ((2.0 * c3 * (1.0 + np.log(T))) / (c2 * T)) ** (1.0 / 3.0)
    else:
        raise ValueError("setting must be 'convex' or 'strongly_convex'")

    return float(delta / r)


def _select_radii_for_xi(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    alpha_values: Iterable[float] = (0.0, 1.0),
    R_grid: Iterable[float] | None = None,
    r_scales: Iterable[float] | None = None,
    network_topology: str = "cycle",
) -> Tuple[Tuple[float, float], Dict[float, float]]:
    """Find radii ``(R, r)`` such that ξ < 1 for all ``alpha_values``."""
    if R_grid is None:
        R_grid = np.linspace(10.0, 1.0, num=19)  # 10.0, 9.5, ..., 1.0
    if r_scales is None:
        r_scales = (1.0, 0.75, 0.5)

    max_abs_b = float(np.abs(y).max(initial=0.0))
    dim = X.shape[1]

    base_adj, _, rho_value, _ = _build_network_topology(network_topology, num_nodes)
    print("rho_value: ", rho_value)
    print("base_adj: ", base_adj)

    for R in R_grid:
        # for scale in r_scales:
        scale = 1.0
        r = R * scale
        feasible = True
        xi_map: Dict[float, float] = {}
        for alpha_reg in alpha_values:
            setting = "strongly_convex" if alpha_reg > 0 else "convex"
            L_f, C = _compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b)
            print("R: ", R)
            print("alpha: ", alpha_reg)
            print("max_abs_b: ", max_abs_b)
            print("L_f: ", L_f)
            print("C: ", C)
            xi = _compute_xi(
                setting=setting,
                T=T,
                dim=dim,
                num_nodes=num_nodes,
                R=R,
                r=r,
                L_f=L_f,
                C=C,
                rho=rho_value,
                alpha=alpha_reg if setting == "strongly_convex" else None,
            )
            xi_map[alpha_reg] = xi
            if xi >= 1.0:
                feasible = False
                break

        if feasible:
            return (float(R), float(r)), xi_map

    raise ValueError(
        "Unable to find radii R and r such that xi < 1 for all provided alpha_values. "
        "Consider expanding the search grid."
    )


# ============================================================
# Main experiment
# ============================================================

def run_bdoo_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int = 20000,
    num_nodes: int = 8,
    R: float = 10.0,
    r: float | None = None,
    alpha_reg: float = 0.0,
    network_topology: str = "cycle",
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Apply BDOO (Algorithm 2) to the linear regression task.

    Returns
    -------
    w_hat : (d,)
        Global model (average across nodes and time).
    X_hist : (T, n, d)
        Model history.
    losses : array-like
        Losses returned by ``run_algorithm2_bandit_paper_params`` (as-is).
    mse : float
        MSE on the full dataset using w_hat (no reg term in MSE).
    avg_loss_per_round : float
        (1/T) Σ_t (1/n) Σ_i f_t^i(x_{i,t}) using your specified loss.
    avg_regret_per_round : float
        Average regret normalized by T against best fixed x in ||x||<=R.
    """
    if rng is None:
        rng = np.random.default_rng()

    if r is None:
        r = R

    n_samples, dim = X.shape

    # Pre-sample which data point each node observes at every round
    samples = rng.integers(0, n_samples, size=(T, num_nodes))
    loss_oracle = _make_loss_oracle(X, y, samples, alpha_reg)

    max_abs_b = float(np.abs(y).max(initial=0.0))
    L_f, C = _compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b)

    # Graph connectivity and consensus parameter
    base_adj, _, rho_value, a = _build_network_topology(network_topology, num_nodes)

    setting = "strongly_convex" if alpha_reg > 0 else "convex"

    X_hist, losses = run_algorithm2_bandit_paper_params(
        loss_oracle=loss_oracle,
        T=T,
        num_nodes=num_nodes,
        dim=dim,
        R=R,
        r=r,
        L_f=L_f,
        C=C,
        rho=rho_value,
        alpha=alpha_reg if setting == "strongly_convex" else None,
        setting=setting,
        a=a,
        base_adj=base_adj,
        graph_mode="cycle",
        rng=rng,
    )

    # Compute average loss per round and average regret per round (normalized by T)
    avg_loss_per_round, avg_regret_per_round = compute_avg_loss_and_avg_regret_per_round(
        X_hist=X_hist,
        samples=samples,
        X=X,
        y=y,
        alpha_reg=alpha_reg,
        R=R,
        use_pre_update_iterate=False,
    )

    # Global model: average across nodes and time
    w_hat = X_hist.mean(axis=(0, 1))

    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))

    return w_hat, X_hist, losses, mse, avg_loss_per_round, avg_regret_per_round


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    print(f"Loaded bodyfat dataset: {X.shape[0]} samples, {X.shape[1]} features")

    T = 100000
    network_topology = "grid3x3"  # "grid3x3" (num_nodes=9) or "cube" (num_nodes=8) or "cycle"
    num_nodes = 9 if network_topology == "grid3x3" else 8

    R = 10.
    r = 10.

    # radii, xi_map = _select_radii_for_xi(
    #     X,
    #     y,
    #     T=T,
    #     num_nodes=num_nodes,
    #     network_topology=network_topology,
    # )
    # R, r = radii
    # print(f"Selected radii: R={R:.3f}, r={r:.3f}")
    # for alpha_reg, xi_val in xi_map.items():
    #     print(f"  alpha_reg={alpha_reg}: xi={xi_val:.6f}")

    # Convex (alpha_reg = 0) and strongly convex (alpha_reg = 1) cases
    for alpha_reg in (0.0, 1.0):
        w_hat, X_hist, losses, mse, avg_loss, avg_regret = run_bdoo_linear_regression(
            X,
            y,
            T=T,
            num_nodes=num_nodes,
            R=R,
            r=r,
            alpha_reg=alpha_reg,
            network_topology=network_topology,
            rng=np.random.default_rng(42),
        )

        print("=" * 60)
        print(f"alpha_reg = {alpha_reg}")
        print(f"Final model shape: {w_hat.shape}")
        print(f"Mean squared error (full dataset): {mse:.6f}")
        print(f"Loss history shape (returned by algo): {np.shape(losses)}")
        print(f"Mean(losses) returned by algo: {float(np.mean(losses)):.6f}")
        print(f"Avg loss per round (your f_t^i, averaged over learners): {avg_loss:.6f}")
        print(f"Avg regret per round (normalized by T): {avg_regret:.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
