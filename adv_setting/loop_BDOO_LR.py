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

The implementation samples data uniformly with replacement for each node and
time, computes the Lipschitz and loss bounds used by BDOO, and reports the
resulting mean-squared error on the full dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler

from optimization_utils.BanditLogisticRegression import (
    _build_cycle_graph,
    _build_gossip_matrix,
    _compute_max_degree,
    run_algorithm2_bandit_paper_params,
)


def _load_bodyfat_dataset(dataset_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the bodyfat dataset.

    Parameters
    ----------
    dataset_path : Path or str
        Path to the LIBSVM formatted bodyfat dataset file.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 14)
        Scaled features with per-feature MaxAbs scaling to [-1, 1] and
        per-sample `\ell_2` normalization.
    y : np.ndarray, shape (n_samples,)
        Target values as a dense array.
    """

    X_sparse, y = load_svmlight_file(str(dataset_path))
    X = X_sparse.toarray()

    X = MaxAbsScaler().fit_transform(X)
    X = preprocessing.normalize(X, norm="l2")

    return X.astype(float), y.astype(float)


def _make_loss_oracle(
    X: np.ndarray,
    y: np.ndarray,
    samples: np.ndarray,
    alpha_reg: float,
) -> Callable[[int, int, np.ndarray], float]:
    """Create the loss oracle f(i, t, x) used by BDOO.

    The oracle samples data points according to ``samples`` and evaluates the
    (regularized) squared loss at the provided query point ``x``.
    """

    def oracle(i: int, t: int, x: np.ndarray) -> float:
        idx = samples[t - 1, i]
        a_i = X[idx]
        b_i = y[idx]
        residual = float(np.dot(a_i, x) - b_i)
        loss = residual * residual
        if alpha_reg:
            loss += 0.5 * alpha_reg * float(np.dot(x, x))
        return loss

    return oracle


def _compute_bounds(R: float, alpha_reg: float, max_abs_b: float) -> Tuple[float, float]:
    """Compute Lipschitz (L_f) and loss bound (C) for squared loss.

    With ``||a||_2 <= 1`` and ``||x||_2 <= R``:

    * |a^T x| <= R
    * |(a^T x - b)| <= R + |b|
    * ||grad|| <= 2 (R + |b|) + alpha_reg * R
    * f(x) <= (R + |b|)^2 + 0.5 * alpha_reg * R^2
    """

    lip = 2.0 * (R + max_abs_b) + alpha_reg * R
    loss_bound = (R + max_abs_b) ** 2 + 0.5 * alpha_reg * (R**2)
    return lip, loss_bound


def _build_grid3x3_graph(num_nodes: int) -> np.ndarray:
    """Return the adjacency for a 3×3 grid network (9 nodes).

    Nodes are labeled row-wise from 0 to 8. Connectivity is 4-neighbor where
    applicable (up, down, left, right), without self-loops.
    """

    if num_nodes != 9:
        raise ValueError("3x3 grid topology requires exactly 9 nodes.")

    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
    rows, cols = 3, 3
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if r > 0:
                up = (r - 1) * cols + c
                adj[idx, up] = True
                adj[up, idx] = True
            if r < rows - 1:
                down = (r + 1) * cols + c
                adj[idx, down] = True
                adj[down, idx] = True
            if c > 0:
                left = r * cols + (c - 1)
                adj[idx, left] = True
                adj[left, idx] = True
            if c < cols - 1:
                right = r * cols + (c + 1)
                adj[idx, right] = True
                adj[right, idx] = True

    return adj


def _build_cube_graph(num_nodes: int) -> np.ndarray:
    """Return the adjacency for an 8-node cube (3-regular) network."""

    if num_nodes != 8:
        raise ValueError("Cube topology requires exactly 8 nodes.")

    adj = np.zeros((num_nodes, num_nodes), dtype=bool)

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    for i, j in edges:
        adj[i, j] = True
        adj[j, i] = True

    return adj


def _build_network_topology(
    topology: str, num_nodes: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Construct adjacency, gossip matrix, degree-weight, and ρ for a topology."""

    if topology == "cycle":
        base_adj = _build_cycle_graph(num_nodes)
    elif topology in {"grid", "grid3x3"}:
        base_adj = _build_grid3x3_graph(num_nodes)
    elif topology in {"cube", "hypercube"}:
        base_adj = _build_cube_graph(num_nodes)
    else:
        raise ValueError("Unsupported topology. Use 'cycle', 'grid3x3', or 'cube'.")

    a = 1.0 / (1.0 + _compute_max_degree(base_adj))
    gossip_matrix = _build_gossip_matrix(base_adj, a)
    eigenvalues = np.linalg.eigvals(gossip_matrix)
    eigenvalues_sorted = np.sort(np.real(eigenvalues))[::-1]
    rho_value = float(eigenvalues_sorted[1])

    return base_adj, gossip_matrix, rho_value, a


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
        delta = np.sqrt(c1 / c2) * (T ** (-0.25))
    elif setting == "strongly_convex":
        if alpha is None or alpha <= 0:
            raise ValueError("alpha must be positive for strongly convex settings")
        c3 = (dim * (C**2) / (2.0 * alpha)) * (
            1.0 + 6.0 * rho * (1.0 + np.sqrt(num_nodes)) / (1.0 - rho)
        )
        c2 = 2.0 * (L_f + C / r)
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
    """Find radii ``(R, r)`` such that ξ < 1 for all ``alpha_values``.

    The search proceeds over a grid of candidate outer radii ``R`` and inner
    radii ``r = scale * R``. The first feasible pair is returned alongside the
    corresponding ``xi`` values keyed by ``alpha_reg``.

    Parameters
    ----------
    network_topology : {"cycle", "grid3x3", "cube"}
        Graph structure used to compute the connectivity term ``ρ``.
    """

    if R_grid is None:
        R_grid = np.linspace(10.0, 1.0, num=19)  # 10.0, 9.5, ..., 1.0
    if r_scales is None:
        r_scales = (1.0, 0.75, 0.5)

    max_abs_b = float(np.abs(y).max(initial=0.0))
    dim = X.shape[1]

    base_adj, _, rho_value, _ = _build_network_topology(network_topology, num_nodes)

    for R in R_grid:
        for scale in r_scales:
            r = R * scale
            feasible = True
            xi_map: Dict[float, float] = {}
            for alpha_reg in alpha_values:
                setting = "strongly_convex" if alpha_reg > 0 else "convex"
                L_f, C = _compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply BDOO (Algorithm 2) to the linear regression task.

    Parameters
    ----------
    X, y : array-like
        Preprocessed dataset.
    T : int
        Time horizon (number of rounds).
    num_nodes : int
        Number of learners in the communication network.
    R : float
        Radius of the feasible `\ell_2` ball.
    r : float, optional
        Inner ball radius. Defaults to ``R`` when not provided.
    alpha_reg : float
        L2 regularization coefficient (0 for convex, 1 for strongly convex
        experiments as described in the implementation details).
    network_topology : {"cycle", "grid3x3", "cube"}
        Communication graph used by the learners. The 3×3 grid option requires
        ``num_nodes == 9`` while the cube requires ``num_nodes == 8``.
    rng : np.random.Generator, optional
        Random generator for reproducibility.
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

    # Global model: average across nodes and time
    w_hat = X_hist.mean(axis=(0, 1))

    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))

    return w_hat, X_hist, losses, mse


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    print(f"Loaded bodyfat dataset: {X.shape[0]} samples, {X.shape[1]} features")
    T = 2000
    network_topology = "cycle"  # set to "grid3x3" for a 9-node grid or "cube" for the 8-node cube
    num_nodes = 9 if network_topology == "grid3x3" else 8

    radii, xi_map = _select_radii_for_xi(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        network_topology=network_topology,
    )
    R, r = radii
    print(f"Selected radii: R={R:.3f}, r={r:.3f}")
    for alpha_reg, xi_val in xi_map.items():
        print(f"  alpha_reg={alpha_reg}: xi={xi_val:.6f}")

    # Convex (alpha_reg = 0) and strongly convex (alpha_reg = 1) cases
    for alpha_reg in (0.0, 1.0):
        w_hat, X_hist, losses, mse = run_bdoo_linear_regression(
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
        print(f"Mean squared error: {mse:.6f}")
        print(f"Loss history shape: {losses.shape}")
        print("=" * 60)


if __name__ == "__main__":
    main()
