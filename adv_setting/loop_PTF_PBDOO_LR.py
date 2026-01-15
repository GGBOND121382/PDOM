# -*- coding: utf-8 -*-
"""
PTF-PBDO^2 (Algorithm 1, convex) with Theorem 1 + Corollary 1 parameters
- Fixes the two common "OCR/typo" mismatches:
  (1) C_tau uses  sqrt(2) * ln(sqrt(14 n))  (NOT sqrt(2 ln(sqrt(14 n))))
  (2) accelerated-consensus theta uses 1/(1 + sqrt(1 - sigma2(P))) (NOT sigma2(P)^2)

Paste this file directly and run inside your repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import math

import numpy as np

# --- your repo utilities (preferred) ---
from optimization_utils.BanditLogisticRegression import (
    _build_network_topology,
    _compute_bounds,
    _load_bodyfat_dataset,
    _make_loss_oracle,
)
from optimization_utils.adv_setting_utils import (
    compute_best_fixed_avg_loss_per_round,
    l2_ball_projection,
    sample_unit_vectors,
    sigma2_of_gossip,
)

Array = np.ndarray


# ============================================================
# Corollary 1 parameter choice (convex case, alpha=0)
#   tau, xi, eta EXACTLY as in your pasted Corollary 1
# ============================================================

def _parse_compute_bounds_output(bounds_out: Any) -> Tuple[float, float]:
    """
    _compute_bounds(...) is expected to provide (L, M) where:
      L = Lipschitz constant
      M = uniform bound |f(x)| <= M
    """
    if isinstance(bounds_out, tuple) and len(bounds_out) == 2:
        L, M = bounds_out
        return float(L), float(M)
    if isinstance(bounds_out, dict):
        # try common keys
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

        return L, M

    raise ValueError(f"Unsupported _compute_bounds output type: {type(bounds_out)}")


def corollary1_params(
    *,
    T: int,
    d: int,
    n: int,
    epsilon: float,
    r: float,
    C_norm: float,        # ||C||_2; for l2-ball radius R => C_norm = R
    P: Array,
    L: float,
    M: float,
) -> Tuple[int, float, float, float]:
    r"""
    Corollary 1 (convex, alpha=0) parameter choice.

    Returns:
      (tau, xi, eta, C_tau)

    Your pasted Corollary 1:
      tau = max{  d^{4/3} T^{1/3} / L^{4/3} * ( M(M+Lr)/(eps r ||C||_2) )^{2/3},  C_tau }
      xi  = T^{-1/4} * sqrt( d M r ||C||_2 (eps sqrt(tau) + d) / ( (M+Lr) eps sqrt(tau) ) )
      eta = ||C||_2 / sqrt( T( d^2 M^2/xi^2 + d^4 M^2/(tau xi^2 eps^2) + tau L^2 ) )

    And the mixing lower bound (from your proof step):
      C_tau = sqrt(2) * ln(sqrt(14 n)) / ((sqrt(2)-1) * sqrt(1 - sigma2(P)))
    """
    if T <= 0 or d <= 0 or n <= 0:
        raise ValueError("Need T, d, n > 0.")
    if epsilon <= 0:
        raise ValueError("Need epsilon > 0.")
    if r <= 0 or C_norm <= 0:
        raise ValueError("Need r, ||C||_2 > 0.")
    if L <= 0 or M <= 0:
        raise ValueError("Need L, M > 0.")

    sig2 = float(sigma2_of_gossip(P))
    gap = 1.0 - sig2
    gap = max(gap, 1e-12)

    # --- FIX #1: C_tau must be sqrt(2) * ln(sqrt(14n)) / ((sqrt(2)-1) * sqrt(1-sigma2(P)))
    C_tau = (math.sqrt(2.0) * math.log(math.sqrt(14.0 * n))) / (
        (math.sqrt(2.0) - 1.0) * math.sqrt(gap)
    )

    tau_expr = (d ** (4.0 / 3.0)) * (T ** (1.0 / 3.0)) / (L ** (4.0 / 3.0))
    tau_expr *= ((M * (M + L * r)) / (epsilon * r * C_norm)) ** (2.0 / 3.0)

    tau = int(math.ceil(max(tau_expr, C_tau)))
    tau = max(tau, 1)

    xi = (T ** (-1.0 / 4.0)) * math.sqrt(
        (d * M * r * C_norm * (epsilon * math.sqrt(tau) + d))
        / ((M + L * r) * epsilon * math.sqrt(tau))
    )

    if not (xi < r):
        raise ValueError(f"Corollary 1 requires xi < r. Got xi={xi:.6g}, r={r:.6g}.")

    denom = T * (
        (d**2 * M**2) / (xi**2)
        + (d**4 * M**2) / (tau * xi**2 * epsilon**2)
        + tau * (L**2)
    )
    eta = C_norm / math.sqrt(denom)

    return tau, xi, eta, C_tau


# ============================================================
# Algorithm 1 (PTF-PBDO^2, convex) implementation
#   Uses Theorem 1 DP noise scale + accelerated consensus
# ============================================================

@dataclass
class PTF_PBDO2_Convex_Config:
    n: int
    d: int
    T: int
    tau: int
    eta: float
    xi: float
    epsilon: float
    M: float
    P: Array
    R: float
    r: Optional[float] = None
    seed: int = 0
    store_actions: bool = False
    follow_paper_delay: bool = True  # z^0(z)=z(z-1)+g(z-1), z^{-1}(z)=z^{tau-1}(z-1)+g(z-1)


def run_ptf_pbd02_convex(
    loss_fn: Callable[[int, int, Array], float],
    cfg: PTF_PBDO2_Convex_Config,
    proj_C: Optional[Callable[[Array, float], Array]] = None,
) -> Dict[str, Any]:
    """
    Implements Algorithm 1 (convex) with:
      - played actions: x_tilde = x + xi * nu
      - one-point estimator: rho = (d/xi) f(x_tilde) nu
      - accelerated consensus inside each batch (z>=2):
            z^{k+1} = (1+theta) P z^k - theta z^{k-1}
      - DP noise at end of batch (Theorem 1):
            b ~ Lap^d( 2 d^{3/2} M / (epsilon * xi) )
      - FTRL update (l2-ball closed form):
            x = Proj_{C_xi}( -eta * z )
    """
    n, d, T, tau = cfg.n, cfg.d, cfg.T, cfg.tau
    if cfg.P.shape != (n, n):
        raise ValueError("P must be (n,n).")

    if tau <= 0:
        raise ValueError("tau must be positive.")
    if tau > T:
        tau = T
    Z = int(math.ceil(T / tau))

    r = cfg.r if cfg.r is not None else cfg.R
    if cfg.xi >= r:
        raise ValueError(f"Need xi < r to ensure x+xi*nu stays in C. Got xi={cfg.xi}, r={r}.")

    proj = proj_C if proj_C is not None else l2_ball_projection
    R_xi = (1.0 - cfg.xi / r) * cfg.R

    sig2 = float(sigma2_of_gossip(cfg.P))
    # --- FIX #2: theta = 1/(1 + sqrt(1 - sigma2(P))) (NOT 1 - sigma2(P)^2)
    theta = 1.0 / (1.0 + math.sqrt(max(1e-12, 1.0 - sig2)))

    rng = np.random.default_rng(cfg.seed)

    # State variables
    x = np.zeros((n, d), dtype=float)              # x_i(z)
    z_vec = np.zeros((n, d), dtype=float)          # z_i(z)
    z_tau_minus_1 = np.zeros((n, d), dtype=float)  # z_i^{tau-1}(z)

    g_prev = np.zeros((n, d), dtype=float)         # g_i(z-1)

    x_batches = np.zeros((Z + 1, n, d), dtype=float)
    z_batches = np.zeros((Z + 1, n, d), dtype=float)
    g_batches = np.zeros((Z, n, d), dtype=float)
    actions = np.zeros((T, n, d), dtype=float) if cfg.store_actions else None

    x_batches[0] = x
    z_batches[0] = z_vec

    # Theorem 1 DP noise scale:
    lap_scale = (2.0 * (d ** 1.5) * cfg.M) / (cfg.epsilon * cfg.xi)

    loss_sum = 0.0
    t_global = 0

    for z in range(1, Z + 1):
        batch_len = min(tau, T - t_global)
        if batch_len <= 0:
            break

        # Initialize accelerated-consensus state for z>=2
        if z >= 2:
            if cfg.follow_paper_delay:
                z_k = z_vec + g_prev
                z_km1 = z_tau_minus_1 + g_prev
            else:
                z_k = z_vec.copy()
                z_km1 = z_tau_minus_1.copy()

        grad_sum = np.zeros((n, d), dtype=float)

        for _ in range(batch_len):
            nus = sample_unit_vectors(rng, n, d)   # each row unit vector
            x_tilde = x + cfg.xi * nus             # played action

            if cfg.store_actions and actions is not None:
                actions[t_global] = x_tilde

            losses = np.empty((n,), dtype=float)
            for i in range(n):
                losses[i] = float(loss_fn(t_global + 1, i, x_tilde[i]))

            loss_sum += float(np.sum(losses))

            # one-point estimator
            rho = (d / cfg.xi) * losses[:, None] * nus
            grad_sum += rho

            # accelerated consensus within the batch for z>=2
            if z >= 2:
                z_kp1 = (1.0 + theta) * (cfg.P @ z_k) - theta * z_km1
                z_km1, z_k = z_k, z_kp1

            t_global += 1

        # DP noise at end of minibatch (Theorem 1)
        noise = rng.laplace(loc=0.0, scale=lap_scale, size=(n, d))
        g_curr = grad_sum + noise
        g_batches[z - 1] = g_curr

        # set z_i(z) = z_i^{tau}(z) (for z>=2)
        if z >= 2:
            z_vec = z_k
            z_tau_minus_1 = z_km1

        z_batches[z] = z_vec

        # FTRL update (closed form for l2-ball C_xi)
        x = proj(-cfg.eta * z_vec, R_xi)
        x_batches[z] = x

        g_prev = g_curr

    out: Dict[str, Any] = {
        "x_batches": x_batches,
        "z_batches": z_batches,
        "g_batches": g_batches,
        "x_final": x,
        "theta": float(theta),
        "sigma2": float(sig2),
        "laplace_scale": float(lap_scale),
        "loss_sum": float(loss_sum),
        "avg_loss_per_round": float(loss_sum) / float(n * T),
        "tau": int(tau),
        "xi": float(cfg.xi),
        "eta": float(cfg.eta),
    }
    if cfg.store_actions and actions is not None:
        out["actions"] = actions
    return out


# ============================================================
# Linear regression on bodyfat dataset (wrapper)
#   Uses Corollary 1 parameter choice ALWAYS (convex alpha=0)
# ============================================================

def run_ptf_pbd02_linear_regression_corollary1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int = 20000,
    num_nodes: int = 8,
    epsilon: float = 1.0,
    R: float = 10.0,
    r: float | None = None,
    alpha_reg: float = 0.0,              # MUST be 0 for Corollary 1 (convex)
    network_topology: str = "cycle",
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict[str, Any], float, float, float]:
    """
    Returns:
      w_hat, res, mse, avg_loss_per_round, avg_regret_per_round
    """
    if rng is None:
        rng = np.random.default_rng()
    if r is None:
        r = R
    if alpha_reg != 0.0:
        raise ValueError("Corollary 1 is for convex case (alpha=0). Set alpha_reg=0.0.")

    n_samples, dim = X.shape

    # Stream assignment (a_{i,t}, b_{i,t})
    samples = rng.integers(0, n_samples, size=(T, num_nodes))
    loss_oracle = _make_loss_oracle(X, y, samples, alpha_reg)

    def loss_fn(t: int, i: int, x: np.ndarray) -> float:
        # note: loss_oracle signature is (i, t, x)
        return float(loss_oracle(i, t, x))

    max_abs_b = float(np.abs(y).max(initial=0.0))
    L, M = _parse_compute_bounds_output(_compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b))

    _, gossip_matrix, _, _ = _build_network_topology(network_topology, num_nodes)

    # Corollary 1 params
    C_norm = R  # for l2-ball C={||x||<=R}, ||C||_2 = R
    tau_c1, xi_c1, eta_c1, C_tau = corollary1_params(
        T=T,
        d=dim,
        n=num_nodes,
        epsilon=epsilon,
        r=float(r),
        C_norm=float(C_norm),
        P=gossip_matrix,
        L=float(L),
        M=float(M),
    )

    cfg = PTF_PBDO2_Convex_Config(
        n=num_nodes,
        d=dim,
        T=T,
        tau=tau_c1,
        eta=eta_c1,
        xi=xi_c1,
        epsilon=epsilon,
        M=float(M),
        P=gossip_matrix,
        R=R,
        r=float(r),
        seed=0,
        store_actions=False,
        follow_paper_delay=True,
    )

    res = run_ptf_pbd02_convex(loss_fn, cfg)

    # Global model (same aggregation style as your original code)
    x_batches = res["x_batches"]
    w_hat = x_batches.mean(axis=(0, 1))

    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))

    avg_loss_per_round = float(res["avg_loss_per_round"])

    best_fixed_avg = compute_best_fixed_avg_loss_per_round(
        samples=samples,
        X=X,
        y=y,
        alpha_reg=alpha_reg,
        R=R,
    )

    avg_regret_per_round = float(avg_loss_per_round - best_fixed_avg)

    res["best_fixed_avg_loss_per_round"] = float(best_fixed_avg)
    res["avg_regret_per_round"] = float(avg_regret_per_round)

    # Store Corollary-1 constants for transparency
    res["L_lipschitz"] = float(L)
    res["M_loss_bound"] = float(M)
    res["C_tau_lower_bound"] = float(C_tau)

    return w_hat, res, mse, avg_loss_per_round, avg_regret_per_round


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
    alpha_reg = 0.0  # MUST be 0 for Corollary 1 (convex)

    w_hat, res, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_corollary1(
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
    print("-" * 60)
    print("Corollary-1 parameters used:")
    print(f"tau              = {res['tau']}")
    print(f"xi               = {res['xi']:.6g}")
    print(f"eta              = {res['eta']:.6g}")
    print(f"C_tau (lower bd) = {res['C_tau_lower_bound']:.6g}")
    print(f"L (Lipschitz)    = {res['L_lipschitz']:.6g}")
    print(f"M (loss bound)   = {res['M_loss_bound']:.6g}")
    print("-" * 60)
    print(f"Final model shape: {w_hat.shape}")
    print(f"Mean squared error: {mse:.6f}")
    print(f"g_batches shape: {res['g_batches'].shape}")
    print(f"Mean(g_batches): {float(np.mean(res['g_batches'])):.6f}")
    print(f"Avg loss per round (normalized by nT):  {avg_loss:.6f}")
    print(f"Best fixed avg loss per round (C ball): {res['best_fixed_avg_loss_per_round']:.6f}")
    print(f"Avg regret per round (normalized):      {avg_regret:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
