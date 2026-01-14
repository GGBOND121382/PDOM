from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np

from optimization_utils.BanditLogisticRegression import (
    _build_network_topology,
    _compute_bounds,
    _load_bodyfat_dataset,
    _make_loss_oracle,
)
from optimization_utils.adv_setting_utils import (
    compute_avg_loss_and_avg_regret_per_round,
    proj_l2_ball,
    sample_unit_vectors,
)


# ============================================================
# Helpers
# ============================================================

def build_row_stochastic_W_from_in_neighbors(n: int, in_neighbors: list[list[int]]) -> np.ndarray:
    """
    Build a row-stochastic weight matrix W where w_ij > 0 only if j is in-neighbor of i (including self),
    and each row sums to 1: w_ij = 1/|N_in(i)| for j in N_in(i) else 0.

    Paper convention: w_ij corresponds to edge (j -> i), i receives from j.
    """
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = in_neighbors[i]
        if len(nbrs) == 0:
            raise ValueError(f"in_neighbors[{i}] is empty; include self-loop to satisfy w_ii>0.")
        w = 1.0 / len(nbrs)
        for j in nbrs:
            W[i, j] = w

    if np.any(W < 0):
        raise ValueError("W has negative entries.")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("W is not row-stochastic (rows must sum to 1).")
    return W


def _estimate_theta_from_W(W: np.ndarray, K: int, z_floor: float = 1e-12) -> float:
    """
    Conservative theta estimate via:
      Z_{t+1} = W Z_t, Z_0 = I,  theta ≈ 1 / min_{s<=K,i} Z_{ii,s}.
    """
    n = W.shape[0]
    Z = np.eye(n, dtype=float)
    min_diag = float(np.min(np.diag(Z)))
    for _ in range(K):
        Z = W @ Z
        min_diag = min(min_diag, float(np.min(np.diag(Z))))
    min_diag = max(min_diag, z_floor)
    return 1.0 / min_diag


def _left_perron_vector_row_stochastic(W: np.ndarray, iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """
    For row-stochastic W, returns psi (normalized, nonnegative) such that psi^T W = psi^T.
    Uses power iteration on W^T.
    """
    n = W.shape[0]
    v = np.ones(n, dtype=float) / n
    for _ in range(iters):
        v_next = W.T @ v
        s = float(np.sum(v_next))
        if s <= 0:
            v_next = np.abs(v_next)
            s = float(np.sum(v_next)) + 1e-12
        v_next /= (s + 1e-12)
        if np.linalg.norm(v_next - v, ord=1) < tol:
            v = v_next
            break
        v = v_next
    v = np.maximum(v, 0.0)
    v /= (float(np.sum(v)) + 1e-12)
    return v


def _estimate_lambda_and_C(W: np.ndarray, psi: np.ndarray, K: int = 300) -> Tuple[float, float]:
    """
    Estimate (lambda, C) for a bound like:
      max_ij |(W^t)_{ij} - psi_j| <= C * lambda^t.

    - lambda: 2nd largest eigenvalue magnitude of W (excluding ~1).
    - C: empirical max over t<=K of (max_ij diff) / lambda^t.
    """
    eigvals = np.linalg.eigvals(W)
    mags = np.sort(np.abs(eigvals))[::-1]

    lam = 0.0
    for m in mags:
        if abs(m - 1.0) > 1e-8:
            lam = float(m)
            break
    lam = float(np.clip(lam, 0.0, 0.999999))

    n = W.shape[0]
    W_inf = np.ones((n, 1)) @ psi.reshape(1, n)

    if lam <= 1e-12:
        return 0.0, 1.0

    Wt = np.eye(n, dtype=float)
    best = 0.0
    for t in range(1, K + 1):
        Wt = Wt @ W
        diff = float(np.max(np.abs(Wt - W_inf)))
        best = max(best, diff / (lam ** t + 1e-18))
    C = max(best, 1.0)
    return lam, C


def _estimate_subgradient_bound_L_for_ridge_ls(X: np.ndarray, y: np.ndarray, R: float, alpha_reg: float) -> float:
    """
    For per-sample loss:
        f(x) = (a^T x - b)^2 + (alpha_reg/2)*||x||^2
    gradient:
        ∇f(x) = 2(a^T x - b)*a + alpha_reg*x

    If ||x||<=R, then |a^T x - b| <= ||a|| R + |b|,
    so ||∇f(x)|| <= 2||a|| (||a|| R + |b|) + alpha_reg R.
    """
    a_max = float(np.linalg.norm(X, axis=1).max(initial=0.0))
    b_max = float(np.abs(y).max(initial=0.0))
    return 2.0 * a_max * (a_max * R + b_max) + float(alpha_reg) * R


# Average regret helpers are imported from optimization_utils.adv_setting_utils.


# ============================================================
# Algorithm 1 (DP + OPBF)
# ============================================================

def algorithm1_dp_opbf(
    W: np.ndarray,
    loss_fn: Callable[[int, int, np.ndarray], float],
    T: int,
    alpha: Callable[[int], float],
    delta: float,
    sigma: Callable[[int], float],
    x0: np.ndarray,
    proj_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    radius: Optional[float] = None,
    seed: int = 0,
    z_diag_floor: float = 1e-12,
    return_history: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Algorithm 1 in the paper (row-stochastic directed network):
      y_{i,t} = x_{i,t} + eta_{i,t}, eta Laplace(scale=sigma_t)
      r_t = W y_t
      Z_{t+1} = W Z_t, Z_0 = I (so z_{ii,t} is Z[i,i])
      g̃_{i,t} = (d/delta) f_{i,t}(x_{i,t} + delta u_{i,t}) u_{i,t}
      x_{i,t+1} = Proj( r_{i,t} - alpha_t * g̃_{i,t} / z_{ii,t} )
    """
    rng = np.random.default_rng(seed)

    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square (n,n) matrix.")
    n = W.shape[0]

    x = np.asarray(x0, dtype=float)
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("x0 must have shape (n, d).")
    d = x.shape[1]

    if proj_fn is None and radius is not None:
        proj_fn = lambda v: proj_l2_ball(v, radius)
        print("radius: ", radius)
    if proj_fn is not None:
        x = np.stack([proj_fn(x[i]) for i in range(n)], axis=0)

    Z = np.eye(n, dtype=float)

    if return_history:
        X_hist = np.zeros((T + 1, n, d), dtype=float)
        X_hist[0] = x

    delta_safe = max(float(delta), 1e-12)

    for t in range(T):
        a = float(alpha(t))
        sig = float(sigma(t))

        eta = rng.laplace(loc=0.0, scale=sig, size=(n, d))
        y = x + eta

        r = W @ y
        Z_next = W @ Z

        u = sample_unit_vectors(rng, n=n, d=d)
        g_tilde = np.zeros((n, d), dtype=float)
        for i in range(n):
            fx = float(loss_fn(i, t, x[i] + delta_safe * u[i]))
            g_tilde[i] = (d / delta_safe) * fx * u[i]

        x_next = np.zeros_like(x)
        for i in range(n):
            z_ii = max(float(Z[i, i]), z_diag_floor)
            v = r[i] - a * (g_tilde[i] / z_ii)
            x_next[i] = proj_fn(v) if proj_fn is not None else v
            # if t % 5000 == 10:
            #     print(np.sum(np.array(x_next[i]) ** 2))

        x, Z = x_next, Z_next

        if return_history:
            X_hist[t + 1] = x

    return (x, X_hist) if return_history else x


# ============================================================
# Linear regression on bodyfat dataset (your wrapper)
# ============================================================

def _build_row_stochastic_W_from_topology(topology: str, num_nodes: int) -> np.ndarray:
    """Build a row-stochastic mixing matrix from an undirected topology."""
    base_adj, _, _, _ = _build_network_topology(topology, num_nodes)
    in_neighbors: list[list[int]] = []
    for i in range(num_nodes):
        neighbors = list(np.where(base_adj[i])[0])
        neighbors.append(i)  # self-loop
        in_neighbors.append(sorted(set(neighbors)))
    return build_row_stochastic_W_from_in_neighbors(num_nodes, in_neighbors)


def _compute_Upsilon_paper(
    *,
    n: int,
    d: int,
    m: int,
    mu: float,
    theta: float,
    Cmix: float,
    lam: float,
    M: float,
    eps: float,
) -> float:
    one_minus_lam = max(1.0 - float(lam), 1e-12)
    eps = max(float(eps), 1e-12)
    mu = max(float(mu), 1e-12)

    n = float(n)
    d = float(d)
    m = float(m)
    theta = float(theta)
    Cmix = float(Cmix)
    M2 = float(M) ** 2

    term1 = 4.0 * np.sqrt(2.0) * (2.0 * n + 1.0) * (n * Cmix * theta) * (d ** 3) * M2 / (eps * one_minus_lam)
    term2 = 4.0 * (m ** 2) * theta * (d ** 2) * M2 / (eps ** 2)
    term3 = 18.0 * np.sqrt(2.0) * (d ** 3) * M2 * n / eps
    term4 = 2.0 * n * Cmix * theta * (d ** 2) * M2 / one_minus_lam
    term5 = 9.0 * theta * (d ** 2) * M2 / 2.0
    return float((term1 + term2 + term3 + term4 + term5) / mu)


def _compute_delta_star_paper(
    *,
    T: int,
    n: int,
    L: float,
    Upsilon: float,
) -> float:
    T = max(int(T), 3)
    n = max(int(n), 1)
    L = max(float(L), 1e-12)
    Upsilon = max(float(Upsilon), 0.0)
    val = (Upsilon * np.log(float(T))) / (2.0 * float(n) * L * float(T))
    return float(max(val, 1e-12) ** (1.0 / 3.0))


def run_pbddo_linear_regression_strong(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int = 20000,
    num_nodes: int = 8,
    epsilon: float = 1.0,
    R: float = 10.0,
    alpha_reg: float = 1.0,
    network_topology: str = "cycle",
    rng: np.random.Generator | None = None,
    delta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Strongly-convex (ridge) linear regression wrapper.

    Returns:
      w_hat, hist, mse, avg_loss_per_round, avg_regret_per_round
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, dim = X.shape

    # data stream assignment
    samples = rng.integers(0, n_samples, size=(T, num_nodes))
    loss_oracle = _make_loss_oracle(X, y, samples, alpha_reg)

    def loss_fn(i: int, t: int, x: np.ndarray) -> float:
        # algorithm1_dp_opbf uses t in {0..T-1}; loss_oracle expects (t+1)
        return float(loss_oracle(i, t + 1, x))

    max_abs_b = float(np.abs(y).max(initial=0.0))
    _, loss_bound = _compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b)

    W = _build_row_stochastic_W_from_topology(network_topology, num_nodes)

    # --- estimate network constants (theta, lambda, C) ---
    theta = _estimate_theta_from_W(W, K=min(T, 5000), z_floor=1e-12)
    psi = _left_perron_vector_row_stochastic(W)
    lam, Cmix = _estimate_lambda_and_C(W, psi, K=300)

    mu = max(float(alpha_reg), 1e-12)
    eps_safe = max(float(epsilon), 1e-12)

    L = _estimate_subgradient_bound_L_for_ridge_ls(X, y, R=R, alpha_reg=alpha_reg)
    M = max(float(loss_bound), 1e-12)

    if delta is None:
        Upsilon = _compute_Upsilon_paper(
            n=num_nodes,
            d=dim,
            m=dim,
            mu=mu,
            theta=theta,
            Cmix=Cmix,
            lam=lam,
            M=M,
            eps=eps_safe,
        )
        delta = _compute_delta_star_paper(T=T, n=num_nodes, L=L, Upsilon=Upsilon)

    delta_safe = max(float(delta), 1e-12)

    delta_safe = min(0.5 * R, delta_safe)

    # P = dM/delta
    P = (float(dim) * M) / delta_safe

    # alpha_t = 1 / (mu * theta * t), t=1..T  -> in 0-based code use (t+1)
    alpha = lambda t: 1.0 / (mu * theta * (t + 1))

    # Δ_t = 2 * theta * alpha_t * sqrt(d) * P
    Delta = lambda t: 2.0 * theta * alpha(t) * np.sqrt(float(dim)) * P

    # sigma_t = Δ_t / epsilon
    sigma = lambda t: Delta(t) / eps_safe

    print("[paper params]")
    print(f"  topology={network_topology}, n={num_nodes}, d={dim}, T={T}")
    print(f"  eps={eps_safe:.6g}, R={R:.6g}, alpha_reg(mu)={mu:.6g}")
    print(f"  theta≈{theta:.6g}, lambda≈{lam:.6g}, C≈{Cmix:.6g}")
    print(f"  L≈{L:.6g}, M≈{M:.6g}")
    print(f"  delta={delta_safe:.6g}  (paper delta* if None was passed)")
    if delta_safe > R:
        print("  [warn] delta > R; this may be too large for OPBF exploration in practice.")
    print(f"  P=dM/delta≈{P:.6g}")
    print(f"  alpha(1)≈{alpha(0):.6g}, Delta(1)≈{Delta(0):.6g}, sigma(1)≈{sigma(0):.6g} (Laplace scale)")

    x0 = np.zeros((num_nodes, dim), dtype=float)
    xT, hist = algorithm1_dp_opbf(
        W=W,
        loss_fn=loss_fn,
        T=T,
        alpha=alpha,
        delta=delta_safe,
        sigma=sigma,
        x0=x0,
        radius=R,
        return_history=True,
        seed=0,
    )

    # ---- average loss / regret normalized by T (averaged over learners) ----
    avg_loss_per_round, avg_regret_per_round = compute_avg_loss_and_avg_regret_per_round(
        X_hist=hist,        # (T+1,n,d)
        samples=samples,    # (T,n)
        X=X,
        y=y,
        alpha_reg=alpha_reg,
        R=R,
        use_pre_update_iterate=True,  # round t uses hist[t]
    )

    w_hat = hist.mean(axis=(0, 1))
    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))
    return w_hat, hist, mse, avg_loss_per_round, avg_regret_per_round


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    print(f"Loaded bodyfat dataset: {X.shape[0]} samples, {X.shape[1]} features")

    T = 100000
    network_topology = "grid3x3"  # "cycle", "grid3x3", or "cube"
    num_nodes = 9 if network_topology == "grid3x3" else 8

    R = 10.0
    epsilon = 1.0
    alpha_reg = 1.0

    w_hat, hist, mse, avg_loss, avg_regret = run_pbddo_linear_regression_strong(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=R,
        alpha_reg=alpha_reg,
        network_topology=network_topology,
        rng=np.random.default_rng(42),
        delta=None,  # None => use paper delta*
    )

    print(w_hat)

    print("=" * 60)
    print(f"alpha_reg = {alpha_reg}")
    print(f"Final model shape: {w_hat.shape}")
    print(f"Mean squared error: {mse:.6f}")
    print(f"History shape (T+1,n,d): {hist.shape}")
    print(f"Avg loss per round (normalized by T):  {avg_loss:.6f}")
    print(f"Avg regret per round (normalized by T): {avg_regret:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
