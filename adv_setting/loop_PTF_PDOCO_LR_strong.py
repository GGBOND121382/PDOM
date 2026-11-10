from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import math

import numpy as np

from optimization_utils.BanditLogisticRegression import (
    _build_network_topology,
    _compute_bounds,
    _load_bodyfat_dataset,
    _make_loss_oracle,
)
from optimization_utils.adv_setting_utils import (
    l2_ball_projection,
    sample_unit_vectors,
    sigma2_of_gossip,
    compute_avg_loss_and_avg_regret_per_round
)

# ============================================================
# IMPORTANT
# ============================================================
# You said `compute_avg_loss_and_avg_regret_per_round` will be manually imported
# from another package. Make sure you import it before running, e.g.:
#
# from <your_package> import compute_avg_loss_and_avg_regret_per_round
#
# The function must implement your corrected regret:
#   Reg(j,T) = Σ_{t=1}^T Σ_{i=1}^N f_{i,t}(x_{j,t})
#              - min_{||x||<=R} Σ_{t=1}^T Σ_{i=1}^N f_{i,t}(x)
#
# and return (avg_loss_per_round, avg_regret_per_round) normalized by T
# (and averaged over learners if that’s how your function is defined).

Array = np.ndarray


# ============================================================
# Utilities
# ============================================================

def _parse_compute_bounds_output(bounds_out: Any) -> Tuple[float, float]:
    """
    Expect _compute_bounds(...) to provide (L, M) where:
      L = Lipschitz constant
      M = uniform bound |f(x)| <= M on C
    """
    if isinstance(bounds_out, tuple) and len(bounds_out) == 2:
        L, M = bounds_out
        return float(L), float(M)

    if isinstance(bounds_out, dict):
        for kL in ("L", "lipschitz", "lip", "L_loss"):
            if kL in bounds_out:
                L = float(bounds_out[kL])
                break
        else:
            raise ValueError(f"Cannot find Lipschitz L in bounds dict keys={list(bounds_out.keys())}")

        for kM in ("M", "loss_bound", "bound", "M_loss"):
            if kM in bounds_out:
                M = float(bounds_out[kM])
                break
        else:
            raise ValueError(f"Cannot find loss bound M in bounds dict keys={list(bounds_out.keys())}")

        return float(L), float(M)

    raise ValueError(f"Unsupported _compute_bounds output type: {type(bounds_out)}")


# ============================================================
# Historical pre-R1-5 parameter schedule (legacy compatibility only).
# ============================================================

def corollary2_params_legacy(
    *,
    T: int,
    d: int,
    n: int,
    epsilon: float,
    alpha: float,
    r: float,
    C_norm: float,   # ||C||_2 ; for l2-ball radius R => C_norm=R
    P: Array,
    L: float,
    M: float,
) -> Tuple[int, float, float]:
    r"""
    Previous pre-R1-5 parameter schedule, retained only for reproducing older runs:

      tau = max{
          alpha^{-1/4} * (L^2/alpha + alpha*||C||_2^2)^(-3/4)
          * sqrt( d^2 (M + Lr) M T / (r epsilon ln T) ),
          C_tau
      }

      xi = ( (r ln T)/(alpha (M+Lr) T) * ( d^2 M^2 + d^4 M^2/(tau epsilon^2) ) )^(1/3)

      C_tau = sqrt(2) * ln(sqrt(14n)) / ((sqrt(2)-1)*sqrt(1 - sigma2(P)))

    Returns:
      (tau_int, xi, C_tau)
    """
    if T <= 1:
        raise ValueError("Corollary 2 uses ln T, require T > 1.")
    if d <= 0 or n <= 0:
        raise ValueError("Need d,n > 0.")
    if epsilon <= 0:
        raise ValueError("Need epsilon > 0.")
    if alpha <= 0:
        raise ValueError("Need alpha > 0 (strong convexity).")
    if r <= 0 or C_norm <= 0:
        raise ValueError("Need r, ||C||_2 > 0.")
    if L <= 0 or M <= 0:
        raise ValueError("Need L, M > 0.")

    lnT = math.log(float(T))

    sig2 = float(sigma2_of_gossip(P))
    gap = max(1e-12, 1.0 - sig2)

    # IMPORTANT: sqrt(2)*ln(sqrt(14n)), NOT sqrt(2*ln(sqrt(14n)))
    C_tau = (math.sqrt(2.0) * math.log(math.sqrt(14.0 * n))) / (
        (math.sqrt(2.0) - 1.0) * math.sqrt(gap)
    )

    term_A = alpha ** (-0.25)
    term_B = (L * L / alpha + alpha * (C_norm ** 2)) ** (-0.75)
    term_C = math.sqrt((d ** 2) * (M + L * r) * M * T / (r * epsilon * lnT))
    tau_expr = term_A * term_B * term_C

    tau = int(math.ceil(max(tau_expr, C_tau)))
    tau = max(tau, 1)
    tau = min(tau, max(1, int(T / 3)))

    xi_inside = (r * lnT) / (alpha * (M + L * r) * T)
    xi_inside *= (d ** 2) * (M ** 2) + (d ** 4) * (M ** 2) / (tau * (epsilon ** 2))
    xi = xi_inside ** (1.0 / 3.0)
    # Fixed feasibility cap avoids discontinuous dyadic shrinking.
    xi = min(xi, 0.9 * r)

    return tau, float(xi), float(C_tau)



def corollary2_params(
    *, T: int, d: int, n: int, epsilon: float, alpha: float,
    r: float, C_norm: float, P: Array, L: float, M: float,
) -> Tuple[int, float, float]:
    r"""Evaluate the R1-5 Corollary 2 parameters as printed in the manuscript.

    The block size is selected first; the exploration radius then depends on
    that block size. The 0.9r cap is the R1-3 feasibility safeguard.
    """
    if not (T > 1 and d > 0 and n > 0 and epsilon > 0 and alpha > 0 and r > 0 and C_norm > 0 and L > 0 and M > 0):
        raise ValueError("Require T>1 and d,n,epsilon,alpha,r,R,L,M>0")
    lnT=math.log(T)
    sig2=float(sigma2_of_gossip(P))
    gap=max(1e-12,1.0-sig2)
    C_tau=math.sqrt(2.0)*math.log(math.sqrt(14.0*n))/((math.sqrt(2.0)-1.0)*math.sqrt(gap))

    tau_expr = (
        alpha**2 * d**4 * M**2 * (M + L*r)**2 * T**2
        / (n * r**2 * epsilon**2 * (L**2 + alpha**2 * C_norm**2)**3 * lnT**2)
    ) ** (1.0 / 4.0)
    tau=int(math.ceil(max(tau_expr,C_tau)))
    if tau > T:
        raise ValueError(f"Corollary 2 requires tau <= T, but computed tau={tau} for T={T}.")

    xi_bandit = (
        d**2 * M**2 * r * lnT
        / (alpha * (M + L*r) * T)
        * (1.0 + d**2 / (n * tau * epsilon**2))
    ) ** (1.0 / 3.0)
    xi_privacy = (
        L * d**2 * M * r * lnT
        / (alpha * epsilon * (M + L*r) * T)
    ) ** (1.0 / 2.0)
    xi=min(max(xi_bandit,xi_privacy),0.9*r)
    return tau,float(xi),float(C_tau)

def fixed_block_params_strong(
    *, T: int, d: int, n: int, epsilon: float, alpha: float,
    r: float, C_norm: float, P: Array, L: float, M: float,
) -> Tuple[int, float, float]:
    """R1-2 Sec. IV-B: tau=ceil(C_tau), xi optimized as in Theorem 2.

    These are the two xi balancing terms of Corollary 2 with the block
    size held fixed. The 0.9r safeguard follows R1-3.
    """
    if not (T > 1 and d > 0 and n > 0 and epsilon > 0 and alpha > 0 and
            r > 0 and C_norm > 0 and L > 0 and M > 0):
        raise ValueError("Require T>1 and d,n,epsilon,alpha,r,R,L,M>0")
    sig2 = float(sigma2_of_gossip(P))
    gap = max(1e-12, 1.0 - sig2)
    C_tau = math.sqrt(2.0) * math.log(math.sqrt(14.0 * n)) / (
        (math.sqrt(2.0) - 1.0) * math.sqrt(gap)
    )
    tau = int(math.ceil(C_tau))
    if tau > T:
        raise ValueError(f"Fixed-block comparison requires ceil(C_tau) <= T: tau={tau}, T={T}")
    lnT = math.log(T)
    xi_bandit = (
        d**2 * M**2 * r * lnT / (alpha * (M + L*r) * T)
        * (1.0 + d**2 / (n * tau * epsilon**2))
    ) ** (1.0 / 3.0)
    xi_privacy = (
        L * d**2 * M * r * lnT / (alpha * epsilon * (M + L*r) * T)
    ) ** 0.5
    xi = min(max(xi_bandit, xi_privacy), 0.9*r)
    return tau, float(xi), float(C_tau)


# ============================================================
# Algorithm 2: Strongly convex PTF-PBDO^2 (as in your pseudocode)
# ============================================================

@dataclass
class PTF_PBDO2_StronglyConvex_Config:
    n: int
    d: int
    T: int
    tau: int
    xi: float
    epsilon: float
    M: float
    alpha: float          # strong convexity parameter
    P: Array              # (n,n) gossip matrix
    R: float              # if C is an l2-ball of radius R (projection)
    r: Optional[float] = None  # interior radius for shrinkage; default r=R for l2-ball
    seed: int = 0
    rng: Optional[np.random.Generator] = None  # shared experiment RNG; seed is fallback for standalone use
    store_actions: bool = False  # store perturbed actions x~_{t,j} so regret can be computed


def run_ptf_pbd02_strongly_convex(
    loss_fn: Callable[[int, int, Array], float],
    cfg: PTF_PBDO2_StronglyConvex_Config,
    proj_C: Optional[Callable[[Array, float], Array]] = None,
) -> Dict[str, Any]:
    """
    Implements Alg. "PTF-PBDO^2 for Strongly Convex Loss" in your text.

    We additionally (optionally) store played actions:
      actions[t, j] = x_tilde = x_j(z) + xi * nu_{t,j}
    which are used as x_{j,t} in your regret definition.
    """
    n, d, T, tau = cfg.n, cfg.d, cfg.T, cfg.tau
    if cfg.P.shape != (n, n):
        raise ValueError("P must be (n,n).")
    if cfg.alpha <= 0:
        raise ValueError("Strongly convex case requires alpha > 0.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if tau > T:
        raise ValueError(f"tau={tau} exceeds T={T}.")

    r = cfg.r if cfg.r is not None else cfg.R
    xi = min(float(cfg.xi), 0.9 * r)
    if xi <= 0.0:
        raise ValueError(f"Need xi > 0. Got xi={xi}.")

    R_xi = (1.0 - xi / r) * cfg.R
    if R_xi <= 0:
        raise ValueError(f"Shrunk radius R_xi must be positive. Got R_xi={R_xi:.6g}.")
    proj = proj_C if proj_C is not None else l2_ball_projection

    sig2 = float(sigma2_of_gossip(cfg.P))
    # your pseudocode uses sqrt(1 - sigma2^2(P))
    theta = 1.0 / (1.0 + math.sqrt(max(1e-12, 1.0 - sig2 * sig2)))

    rng = cfg.rng if cfg.rng is not None else np.random.default_rng(cfg.seed)

    # Laplace noise scale: Lap^d( 2 d^{3/2} M / (epsilon * xi) )
    lap_scale = (2.0 * (d ** 1.5) * cfg.M) / (cfg.epsilon * xi)

    Z = int(math.ceil(T / tau))

    # States
    x = np.zeros((n, d), dtype=float)              # x_i(1)=0 and x_i(2)=0
    z_vec = np.zeros((n, d), dtype=float)          # z_i(z)
    z_tau_minus_1 = np.zeros((n, d), dtype=float)  # z_i^{tau-1}(z)
    d_prev = np.zeros((n, d), dtype=float)         # d_i(z-1)

    x_batches = np.zeros((Z + 2, n, d), dtype=float)  # x(1)..x(Z+2)
    z_batches = np.zeros((Z + 1, n, d), dtype=float)
    d_batches = np.zeros((Z, n, d), dtype=float)
    actions = np.zeros((T, n, d), dtype=float) if cfg.store_actions else None

    x_batches[0] = x  # x(1)
    x_batches[1] = x  # x(2)
    z_batches[0] = z_vec

    t_global = 0
    for z in range(1, Z + 1):
        batch_len = min(tau, T - t_global)
        if batch_len <= 0:
            break

        if z >= 2:
            z_k = z_vec + d_prev
            z_km1 = z_tau_minus_1 + d_prev

        grad_sum = np.zeros((n, d), dtype=float)

        for _ in range(batch_len):
            nus = sample_unit_vectors(rng, n, d)
            x_tilde = x + xi * nus

            if cfg.store_actions and actions is not None:
                actions[t_global] = x_tilde  # actions[t, j] = x_{j,t} for regret

            losses = np.empty((n,), dtype=float)
            for i in range(n):
                losses[i] = float(loss_fn(t_global + 1, i, x_tilde[i]))

            rho = (d / xi) * losses[:, None] * nus
            grad_sum += rho

            if z >= 2:
                z_kp1 = (1.0 + theta) * (cfg.P @ z_k) - theta * z_km1
                z_km1, z_k = z_k, z_kp1

            t_global += 1

        noise = rng.laplace(loc=0.0, scale=lap_scale, size=(n, d))

        # d_i(z) = sum_{t in batch} (rho_{t,i} - alpha x_i(z)) + noise
        d_curr = grad_sum - (batch_len * cfg.alpha) * x + noise
        d_batches[z - 1] = d_curr

        if z >= 2:
            z_vec = z_k
            z_tau_minus_1 = z_km1
        z_batches[z] = z_vec

        # x_i(z+1): for z>=2, FTAL closed form; for z=1 keep x(2)=0
        if z >= 2:
            x_unproj = -(1.0 / ((z - 1) * tau * cfg.alpha)) * z_vec
            x_next = proj(x_unproj, R_xi)
        else:
            x_next = x

        x_batches[z + 1] = x_next
        x = x_next
        d_prev = d_curr

    out: Dict[str, Any] = {
        "x_batches": x_batches,
        "z_batches": z_batches,
        "d_batches": d_batches,
        "x_final": x,
        "theta": float(theta),
        "sigma2": float(sig2),
        "laplace_scale": float(lap_scale),
        "tau": int(tau),
        "xi": float(xi),
        "alpha": float(cfg.alpha),
    }
    if cfg.store_actions and actions is not None:
        out["actions"] = actions  # (T,n,d)
    return out


# ============================================================
# Linear regression on Bodyfat dataset (wrapper)
# Uses the selected Corollary 2 schedule (refined by default).
# ============================================================

def run_ptf_pbd02_linear_regression_strong_corollary2(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int = 20000,
    num_nodes: int = 8,
    epsilon: float = 1.0,
    R: float = 10.0,
    r: float | None = None,
    alpha_reg: float = 1.0,
    network_topology: str = "cycle",
    rng: np.random.Generator | None = None,
    parameter_selection: str | None = None,
) -> Tuple[np.ndarray, Dict[str, Any], float, float, float]:
    """
    Returns:
      w_hat, res, mse, avg_loss_per_round, avg_regret_per_round

    avg_loss_per_round and avg_regret_per_round are computed by the external
    compute_avg_loss_and_avg_regret_per_round(...) you will import manually.
    """
    if rng is None:
        rng = np.random.default_rng()
    if r is None:
        r = R
    if alpha_reg <= 0:
        raise ValueError("Strongly convex case requires alpha_reg > 0.")

    n_samples, dim = X.shape

    # Stream assignment (a_{i,t}, b_{i,t}) for each learner i
    samples = rng.integers(0, n_samples, size=(T, num_nodes))
    loss_oracle = _make_loss_oracle(X, y, samples, alpha_reg)

    def loss_fn(t: int, i: int, x: np.ndarray) -> float:
        return float(loss_oracle(i, t, x))

    max_abs_b = float(np.abs(y).max(initial=0.0))
    L, M = _parse_compute_bounds_output(_compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b))

    _, gossip_matrix, _, _ = _build_network_topology(network_topology, num_nodes)

    C_norm = R
    import os
    tuning = (parameter_selection or os.environ.get("PDOM_R15_TUNING", "refined")).lower()
    tuning_fns = {
        "refined": corollary2_params,
        "legacy": corollary2_params_legacy,
        "fixed_block": fixed_block_params_strong,
    }
    if tuning not in tuning_fns:
        raise ValueError(f"Unsupported parameter selection: {tuning}")
    tuning_fn = tuning_fns[tuning]
    tau_c2, xi_c2, C_tau = tuning_fn(
        T=T,
        d=dim,
        n=num_nodes,
        epsilon=epsilon,
        alpha=alpha_reg,
        r=float(r),
        C_norm=float(C_norm),
        P=gossip_matrix,
        L=float(L),
        M=float(M),
    )

    cfg = PTF_PBDO2_StronglyConvex_Config(
        n=num_nodes,
        d=dim,
        T=T,
        tau=tau_c2,
        xi=xi_c2,
        epsilon=epsilon,
        M=float(M),
        alpha=float(alpha_reg),
        P=gossip_matrix,
        R=R,
        r=float(r),
        seed=0,
        rng=rng,
        store_actions=True,  # IMPORTANT: needed by compute_avg_loss_and_avg_regret_per_round
    )

    res = run_ptf_pbd02_strongly_convex(loss_fn, cfg)

    # Aggregate model
    x_batches = res["x_batches"]
    w_hat = x_batches.mean(axis=(0, 1))
    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))

    # ---- Compute avg loss + avg regret via the external function ----
    if "compute_avg_loss_and_avg_regret_per_round" not in globals():
        raise ImportError(
            "compute_avg_loss_and_avg_regret_per_round is not imported. "
            "Please manually import it before running."
        )
    if "actions" not in res:
        raise RuntimeError("actions not found; expected cfg.store_actions=True.")

    avg_loss_per_round, avg_regret_per_round = compute_avg_loss_and_avg_regret_per_round(
        X_hist=res["actions"],  # (T,n,d): use played x_{j,t} (here x_tilde)
        samples=samples,        # (T,n): defines (a_{i,t}, b_{i,t})
        X=X,
        y=y,
        alpha_reg=alpha_reg,
        R=R,
        use_pre_update_iterate=True,  # irrelevant for (T,n,d)
    )

    res["avg_loss_per_round"] = float(avg_loss_per_round)
    res["avg_regret_per_round"] = float(avg_regret_per_round)
    res["L_lipschitz"] = float(L)
    res["M_loss_bound"] = float(M)
    res["C_tau_lower_bound"] = float(C_tau)

    print("=" * 60)
    print(f"network_topology = {network_topology}")
    print(f"num_nodes        = {num_nodes}")
    print(f"T                = {T}")
    print(f"epsilon          = {epsilon}")
    print(f"R, r             = {R}, {r}")
    print(f"alpha            = {alpha_reg}")
    print("-" * 60)
    print(f"Parameters used ({tuning}):")
    print(f"tau              = {res['tau']}")
    print(f"xi               = {res['xi']:.6g}")
    print(f"C_tau (lower bd) = {res['C_tau_lower_bound']:.6g}")
    print(f"L (Lipschitz)    = {res['L_lipschitz']:.6g}")
    print(f"M (loss bound)   = {res['M_loss_bound']:.6g}")
    print(f"theta            = {res['theta']:.6g}  (uses 1 - sigma2(P)^2)")
    print("-" * 60)
    print(f"Final model shape: {w_hat.shape}")
    print(f"Mean squared error: {mse:.6f}")
    print(f"d_batches shape: {res['d_batches'].shape}")
    print(f"Avg loss per round (correct regret loss): {res['avg_loss_per_round']:.6f}")
    print(f"Avg regret per round (normalized by T):  {res['avg_regret_per_round']:.6f}")
    print("=" * 60)

    return w_hat, res, mse, float(avg_loss_per_round), float(avg_regret_per_round)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    print(f"Loaded bodyfat dataset: {X.shape[0]} samples, {X.shape[1]} features")

    T = 100000
    network_topology = "grid3x3"  # "cycle", "grid3x3", or "cube"
    num_nodes = 9 if network_topology == "grid3x3" else 8

    R = 10.0
    r = 10.0
    epsilon = 1.0
    alpha_reg = 1.0

    w_hat, res, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_strong_corollary2(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=R,
        r=r,
        alpha_reg=alpha_reg,
        network_topology=network_topology,
        rng=np.random.default_rng(42),
    )

    print("=" * 60)
    print(f"network_topology = {network_topology}")
    print(f"num_nodes        = {num_nodes}")
    print(f"T                = {T}")
    print(f"epsilon          = {epsilon}")
    print(f"R, r             = {R}, {r}")
    print(f"alpha            = {alpha_reg}")
    print("-" * 60)
    print("Corollary-2 parameters used (as in your text):")
    print(f"tau              = res['tau'] -> {res['tau']}")
    print(f"xi               = res['xi']  -> {res['xi']:.6g}")
    print(f"C_tau (lower bd) = res['C_tau_lower_bound'] -> {res['C_tau_lower_bound']:.6g}")
    print(f"L (Lipschitz)    = res['L_lipschitz'] -> {res['L_lipschitz']:.6g}")
    print(f"M (loss bound)   = res['M_loss_bound'] -> {res['M_loss_bound']:.6g}")
    print(f"theta            = res['theta'] -> {res['theta']:.6g}")
    print("-" * 60)
    print(f"Final model shape: {w_hat.shape}")
    print(f"Mean squared error: {mse:.6f}")
    print(f"d_batches shape: {res['d_batches'].shape}")
    print(f"Avg loss per round (correct regret loss): {avg_loss:.6f}")
    print(f"Avg regret per round (normalized by T):  {avg_regret:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
