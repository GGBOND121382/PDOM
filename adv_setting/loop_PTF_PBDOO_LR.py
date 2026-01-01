import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

Array = np.ndarray


def sample_unit_vectors(rng: np.random.Generator, n: int, d: int) -> Array:
    """Sample n random unit vectors uniformly from S^{d-1} using normal+normalize."""
    v = rng.normal(size=(n, d))
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def l2_ball_projection(X: Array, radius: float) -> Array:
    """Row-wise projection onto an l2 ball of given radius."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    factors = np.minimum(1.0, radius / norms)
    return X * factors


def sigma2_of_gossip(P: Array) -> float:
    """Second largest singular value sigma_2(P)."""
    svals = np.linalg.svd(P, compute_uv=False)
    svals_sorted = np.sort(svals)[::-1]
    if len(svals_sorted) < 2:
        return 0.0
    return float(svals_sorted[1])


@dataclass
class PTF_PBDO2_Convex_Config:
    n: int                 # number of learners
    d: int                 # model dimension
    T: int                 # total rounds
    tau: int               # batch size
    eta: float             # step size in Alg. 1 line 23
    xi: float              # exploration radius
    epsilon: float         # DP budget
    M: float               # bound |f_{t,i}(x)| <= M
    P: Array               # (n,n) gossip matrix, doubly stochastic
    R: float               # radius of C if C is l2-ball (default projection)
    r: Optional[float] = None   # interior radius r in C_xi=(1-xi/r)C; default r=R for l2-ball
    seed: int = 0
    store_actions: bool = False # if True, store perturbed actions x~_{t,i}

    # If True, follow Alg. 1 literally: mixing in batch z uses g(z-1).
    # If False, you can later adapt to a “no-delay” variant (not in the paper).
    follow_paper_delay: bool = True


def run_ptf_pbd02_convex(
    loss_fn: Callable[[int, int, Array], float],
    cfg: PTF_PBDO2_Convex_Config,
    proj_C: Optional[Callable[[Array, float], Array]] = None,
) -> Dict[str, Any]:
    """
    Implements Algorithm 1 (PTF-PBDO^2 for convex loss) from the paper.

    loss_fn(t, i, x) -> scalar loss f_{t,i}(x). Must satisfy |f| <= M on C.
    Default feasible set C is an l2 ball of radius R with projection l2_ball_projection.
    The shrunken set is C_xi = (1 - xi/r) C, implemented as radius R_xi = (1 - xi/r)*R.

    Returns:
      - x_batches: base models x_i(z) at each batch start (including initial), shape (Z+1, n, d)
      - x_final: final base models, shape (n, d)
      - g_batches: noisy batch gradients g_i(z), shape (Z, n, d)
      - z_batches: mixed parameters z_i(z), shape (Z+1, n, d)
      - (optional) actions: perturbed actions x~_{t,i}, shape (T, n, d) if store_actions=True
    """
    n, d, T, tau = cfg.n, cfg.d, cfg.T, cfg.tau
    assert cfg.P.shape == (n, n), "P must be (n,n)"
    assert T % tau == 0, "This implementation assumes T is divisible by tau (as in the paper)."

    r = cfg.r if cfg.r is not None else cfg.R
    if cfg.xi >= r:
        raise ValueError(f"Need xi < r to ensure x + xi*nu stays in C. Got xi={cfg.xi}, r={r}.")

    proj = proj_C if proj_C is not None else l2_ball_projection
    R_xi = (1.0 - cfg.xi / r) * cfg.R

    # Compute theta from sigma2(P) (Alg. 1 line 15)
    sig2 = sigma2_of_gossip(cfg.P)
    theta = 1.0 / (1.0 + np.sqrt(max(1e-12, 1.0 - sig2**2)))

    rng = np.random.default_rng(cfg.seed)

    Z = T // tau  # number of batches

    # States (all are (n,d))
    x = np.zeros((n, d), dtype=float)          # x_i(z) base decision (held fixed within batch z)
    z_vec = np.zeros((n, d), dtype=float)      # z_i(z) mixed accumulator used in FTRL update
    z_tau_minus_1 = np.zeros((n, d), dtype=float)  # stores z_i^{tau-1}(z) for next batch init

    g_prev = np.zeros((n, d), dtype=float)     # g_i(z-1), used when z>=2 (paper’s delay)

    # Logs
    x_batches = np.zeros((Z + 1, n, d), dtype=float)
    z_batches = np.zeros((Z + 1, n, d), dtype=float)
    g_batches = np.zeros((Z, n, d), dtype=float)
    actions = np.zeros((T, n, d), dtype=float) if cfg.store_actions else None

    x_batches[0] = x
    z_batches[0] = z_vec

    # Laplace scale for each coordinate (Alg. 1 line 19)
    lap_scale = (2.0 * (d ** 1.5) * cfg.M) / (cfg.epsilon * cfg.xi)

    t_global = 0
    for z in range(1, Z + 1):
        # --- Initialize accelerated mixing variables for this batch (Alg. 1 lines 5-9)
        if z >= 2:
            if cfg.follow_paper_delay:
                # z_i^0(z) = z_i(z-1) + g_i(z-1)
                z_k = z_vec + g_prev
                # z_i^{-1}(z) = z_i^{tau-1}(z-1) + g_i(z-1)
                z_km1 = z_tau_minus_1 + g_prev
            else:
                # Not in the paper: placeholder if you later want a variant
                z_k = z_vec
                z_km1 = z_tau_minus_1

        grad_sum = np.zeros((n, d), dtype=float)

        # --- Within-batch time loop (Alg. 1 lines 10-18)
        for _ in range(tau):
            nus = sample_unit_vectors(rng, n, d)
            x_tilde = x + cfg.xi * nus
            if cfg.store_actions:
                actions[t_global] = x_tilde

            # Query losses and form bandit estimator rho_{t,i}
            losses = np.empty((n,), dtype=float)
            for i in range(n):
                losses[i] = float(loss_fn(t_global + 1, i, x_tilde[i]))  # t is 1-indexed in paper
            rho = (d / cfg.xi) * losses[:, None] * nus
            grad_sum += rho

            # Accelerated mixing step per time t (Alg. 1 line 15)
            if z >= 2:
                z_kp1 = (1.0 + theta) * (cfg.P @ z_k) - theta * z_km1
                z_km1, z_k = z_k, z_kp1

            t_global += 1

        # --- End of batch: add Laplace noise (Alg. 1 line 19)
        noise = rng.laplace(loc=0.0, scale=lap_scale, size=(n, d))
        g_curr = grad_sum + noise
        g_batches[z - 1] = g_curr

        # --- Save mixing outputs (Alg. 1 lines 20-22)
        if z >= 2:
            z_vec = z_k               # z_i(z) = z_i^{tau}(z)
            z_tau_minus_1 = z_km1     # keep z_i^{tau-1}(z) for next batch
        else:
            # For z=1, Alg. 1 does not set z_i(1) via mixing; keep as initialized (zero).
            pass

        z_batches[z] = z_vec

        # --- FTRL update: x_i(z+1) = argmin_{x in C_xi} <z_i(z), x> + (1/2eta)||x||^2
        # Solution is projection of (-eta z_i(z)) onto C_xi (Alg. 1 line 23).
        x = proj(-cfg.eta * z_vec, R_xi)
        x_batches[z] = x

        # g(z) becomes g_prev for next batch’s initialization (paper’s delay)
        g_prev = g_curr

    out: Dict[str, Any] = {
        "x_batches": x_batches,   # (Z+1, n, d)
        "z_batches": z_batches,   # (Z+1, n, d)
        "g_batches": g_batches,   # (Z, n, d)
        "x_final": x,             # (n, d)
        "theta": theta,
        "sigma2": sig2,
        "laplace_scale": lap_scale,
    }
    if cfg.store_actions:
        out["actions"] = actions  # (T, n, d)
    return out


# -----------------------------
# Example usage (toy)
# -----------------------------
if __name__ == "__main__":
    # Toy loss: squared loss to a drifting target per learner
    def toy_loss(t: int, i: int, x: Array) -> float:
        # target drifts slowly; keep bounded by clipping
        target = np.ones_like(x) * (0.1 * (i + 1))
        val = float(np.sum((x - target) ** 2))
        return float(np.clip(val, 0.0, 10.0))  # enforce boundedness by clipping for the toy example

    n, d = 5, 3
    # Simple ring gossip matrix (doubly stochastic)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1 / 3
        P[i, (i - 1) % n] = 1 / 3
        P[i, (i + 1) % n] = 1 / 3

    cfg = PTF_PBDO2_Convex_Config(
        n=n, d=d, T=200, tau=10,
        eta=0.5, xi=0.1, epsilon=1.0, M=10.0,
        P=P, R=10.0, r=10.0,
        seed=42, store_actions=False,
    )

    res = run_ptf_pbd02_convex(toy_loss, cfg)
    print("x_final:\n", res["x_final"])
    print("theta:", res["theta"], "sigma2:", res["sigma2"])
