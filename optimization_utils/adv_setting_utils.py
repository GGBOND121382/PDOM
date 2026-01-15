from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Sample n random unit vectors uniformly from S^{d-1} using normal+normalize."""
    v = rng.normal(size=(n, d))
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def sample_unit_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Alias for sampling unit vectors (kept for backward compatibility)."""
    return sample_unit_vectors(rng, n, d)


def l2_ball_projection(X: np.ndarray, radius: float) -> np.ndarray:
    """Row-wise projection onto an l2 ball of given radius."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12
    factors = np.minimum(1.0, radius / norms)
    return X * factors


def proj_l2_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """Projection onto an l2 ball (works for vectors or row-wise for matrices)."""
    return l2_ball_projection(x, radius)


def sigma2_of_gossip(P: np.ndarray) -> float:
    """Second largest singular value sigma_2(P)."""
    svals = np.linalg.svd(P, compute_uv=False)
    svals_sorted = np.sort(svals)[::-1]
    if len(svals_sorted) < 2:
        return 0.0
    return float(svals_sorted[1])


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b; fall back to least-squares if singular/ill-conditioned."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _solve_ridge_in_l2_ball(
    G: np.ndarray,
    h: np.ndarray,
    *,
    alpha_reg: float,
    scale: float,
    R: float,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> np.ndarray:
    """
    Solve:
        min_{||x||<=R}  sum (a^T x - b)^2 + sum (alpha/2)||x||^2
    using aggregated G=sum aa^T and h=sum ab over all samples.
    """
    d = G.shape[0]
    I = np.eye(d)
    reg = 0.5 * float(alpha_reg) * float(scale)

    def x_of(lam: float) -> np.ndarray:
        return _safe_solve(G + (reg + lam) * I, h)

    x0 = x_of(0.0)
    if float(np.linalg.norm(x0)) <= R:
        return x0

    lam_lo, lam_hi = 0.0, 1.0
    while float(np.linalg.norm(x_of(lam_hi))) > R:
        lam_hi *= 2.0

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        x_mid = x_of(lam_mid)
        if float(np.linalg.norm(x_mid)) > R:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if lam_hi - lam_lo <= tol * max(1.0, lam_hi):
            break

    return x_of(lam_hi)


def compute_best_fixed_avg_loss_per_round(
    *,
    samples: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha_reg: float,
    R: float,
) -> float:
    """
    Compute:
      min_{||x||<=R} (1/(nT)) sum_{t,i} [ (a_{i,t}^T x - b_{i,t})^2 ] + (alpha/2)||x||^2
    using aggregated G=sum aa^T, h=sum ab, sum_b2 from the sampled stream.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (T,n), got {samples.shape}")

    T, n = samples.shape
    idx_flat = samples.reshape(-1)
    A_all = X[idx_flat]
    b_all = y[idx_flat]

    G = A_all.T @ A_all
    h = A_all.T @ b_all
    sum_b2 = float(b_all @ b_all)

    x_star = _solve_ridge_in_l2_ball(G, h, alpha_reg=alpha_reg, scale=float(n * T), R=R)

    comp_sq_avg = float(x_star @ G @ x_star - 2.0 * h @ x_star + sum_b2) / float(n * T)
    comp_avg = comp_sq_avg + 0.5 * float(alpha_reg) * float(x_star @ x_star)
    return float(comp_avg)


def compute_avg_loss_and_avg_regret_per_round(
    X_hist: np.ndarray,
    samples: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_reg: float,
    R: float,
    use_pre_update_iterate: bool = True,
) -> Tuple[float, float]:
    """
    Compute average loss per round and average regret per round (both normalized by T)
    for squared loss with optional L2 regularization.
    """
    if X_hist.ndim != 3:
        raise ValueError(f"X_hist must be 3D, got shape {X_hist.shape}")

    T_s, n_s = samples.shape

    def _to_Tnd(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == T_s and arr.shape[1] == n_s:
            return arr
        if arr.shape[0] == n_s and arr.shape[1] == T_s:
            return np.transpose(arr, (1, 0, 2))
        if arr.shape[0] == T_s + 1 and arr.shape[1] == n_s:
            return arr[:-1] if use_pre_update_iterate else arr[1:]
        if arr.shape[0] == n_s and arr.shape[1] == T_s + 1:
            tmp = np.transpose(arr, (1, 0, 2))
            return tmp[:-1] if use_pre_update_iterate else tmp[1:]
        raise ValueError(
            "Cannot align X_hist with samples.\n"
            f"  X_hist.shape = {arr.shape}\n"
            f"  samples.shape = {samples.shape}\n"
            "Expected X_hist to be one of: (T,n,d), (n,T,d), (T+1,n,d), (n,T+1,d)."
        )

    X_seq = _to_Tnd(X_hist)
    T, n, d = X_seq.shape

    G = np.zeros((d, d), dtype=float)
    h = np.zeros(d, dtype=float)
    sum_b2 = 0.0
    total_loss = 0.0

    for t in range(T):
        idx = samples[t]
        A_t = X[idx]
        b_t = y[idx]
        x_t = X_seq[t]

        preds = np.einsum("nd,nd->n", A_t, x_t)
        resid = preds - b_t
        sq_sum = float(resid @ resid)
        reg_sum = 0.5 * float(alpha_reg) * float(np.sum(x_t * x_t))
        total_loss += sq_sum + reg_sum

        G += A_t.T @ A_t
        h += A_t.T @ b_t
        sum_b2 += float(b_t @ b_t)

    avg_loss_per_round = total_loss / float(T * n)

    x_star = _solve_ridge_in_l2_ball(G, h, alpha_reg=alpha_reg, scale=float(T * n), R=R)
    comp_sq_avg = float(x_star @ G @ x_star - 2.0 * h @ x_star + sum_b2) / float(T * n)
    comp_avg = comp_sq_avg + 0.5 * float(alpha_reg) * float(x_star @ x_star)

    avg_regret_per_round = float(avg_loss_per_round - comp_avg)
    return float(avg_loss_per_round), float(avg_regret_per_round)
