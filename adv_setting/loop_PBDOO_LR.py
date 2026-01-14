from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

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
    sample_unit_sphere,
)

# ============================================================
# Helpers
# ============================================================

Array = np.ndarray
CostOracle = Callable[[int, int, np.ndarray], float]  # cost_oracle(t, i, x) -> scalar (t is 1-based, i is 0-based)


def proj_box(x: np.ndarray, low: Union[float, np.ndarray], high: Union[float, np.ndarray]) -> np.ndarray:
    """Euclidean projection onto a box [low, high] (elementwise)."""
    return np.minimum(np.maximum(x, low), high)


def build_column_stochastic_A(
    n: int,
    edges: Iterable[Tuple[int, int]],
    add_self_loops: bool = True,
) -> np.ndarray:
    """
    Build the column-stochastic weight matrix A used by push-sum.

    Input edges are directed (sender, receiver).
    We set A[receiver, sender] = 1 / |N_out(sender)| for receiver in N_out(sender),
    where N_out(sender) includes sender itself if add_self_loops=True.
    """
    out_sets: List[set[int]] = [set() for _ in range(n)]
    for s, r in edges:
        out_sets[s].add(r)
    if add_self_loops:
        for j in range(n):
            out_sets[j].add(j)

    A = np.zeros((n, n), dtype=float)
    for sender in range(n):
        outs = sorted(out_sets[sender])
        if not outs:
            A[sender, sender] = 1.0
            continue
        w = 1.0 / len(outs)
        for receiver in outs:
            A[receiver, sender] = w

    colsum = A.sum(axis=0)
    if np.any(np.abs(colsum - 1.0) > 1e-8):
        raise ValueError(f"A is not column-stochastic; max |colsum-1|={np.max(np.abs(colsum-1.0))}")
    return A


def stationary_dist_right_eigvec(A: np.ndarray) -> np.ndarray:
    """
    For column-stochastic A, compute pi >= 0, sum(pi)=1 such that A @ pi = pi.
    """
    w, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, k])
    pi = np.abs(pi)
    pi = pi / (pi.sum() + 1e-12)
    return pi


def lambda2_magnitude(A: np.ndarray) -> float:
    """
    Approximate the paper's mixing parameter via the 2nd largest eigenvalue magnitude of A.
    """
    eigvals = np.linalg.eigvals(A)
    mags = np.sort(np.abs(eigvals))
    if len(mags) < 2:
        return 0.0
    return float(mags[-2])


def lipschitz_bound_squared_loss(X: np.ndarray, y: np.ndarray, R: float, alpha_reg: float) -> float:
    """
    Safe Lipschitz constant bound for f(x) = (a^T x - b)^2 + (alpha_reg/2)||x||^2 over ||x||<=R.
      ||∇f(x)|| <= 2|a^T x - b|*||a|| + alpha_reg||x||
               <= 2 (||a|| R + |b|) ||a|| + alpha_reg R
    """
    a_norm = np.linalg.norm(X, axis=1)
    a_norm_max = float(np.max(a_norm)) if a_norm.size else 0.0
    b_abs_max = float(np.max(np.abs(y))) if y.size else 0.0
    return 2.0 * (a_norm_max * R + b_abs_max) * a_norm_max + alpha_reg * R


# ============================================================
# Algorithm (7) implementation (Section 3.1)
# ============================================================

@dataclass
class Algo7Config:
    n: int
    d: int
    T: int
    delta: float
    epsilon: float
    zeta: float = 1.0
    C: Optional[float] = None                 # bound on |f_{t,i}(x)|
    l_i: Union[float, np.ndarray] = 0.0       # Assumption 5: E||tau_{t,i}||^2 <= l_i
    use_tau: bool = False                     # if True, inject tau (must set l_i consistently)
    tau_std: float = 0.0                      # if use_tau: tau ~ N(0, tau_std^2 I)
    seed: int = 0


def alpha_schedule_zeta_over_sqrt_t(zeta: float) -> Callable[[int], float]:
    # Corollary 1 / Theorem 2-3: alpha_t = zeta / sqrt(t)
    def alpha(t: int) -> float:
        return zeta / np.sqrt(max(t, 1))
    return alpha


def compute_p_hat(cfg: Algo7Config) -> float:
    if cfg.C is None:
        raise ValueError("cfg.C must be provided to compute p_hat.")
    l = np.asarray(cfg.l_i, dtype=float)
    if l.size == 1:
        l = np.full((cfg.n,), float(l))
    return float(np.max(cfg.d * cfg.C / max(cfg.delta, 1e-12) + l))


def compute_sigma_const_from_lemma2(cfg: Algo7Config) -> float:
    """
    Theorem 1 uses sigma(t)=Delta(t)/epsilon.
    Lemma 2 gives Delta(t) <= 2 * sqrt(d) * p_hat, where p_hat = max_i{ dC/delta + l_i }.
    We use the bound to form a constant sigma.
    """
    p_hat = compute_p_hat(cfg)
    Delta_bound = 2.0 * np.sqrt(cfg.d) * p_hat
    return Delta_bound / cfg.epsilon


def prox_map_quadratic_psi(z_over_w: np.ndarray, alpha_t: float, proj_X: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    For psi(x)=1/2||x||^2, prox becomes: argmin_{x in X} <z, x> + (1/alpha) * (1/2)||x||^2 = Proj_X(-alpha z)
    """
    return proj_X(-alpha_t * z_over_w)


def run_algorithm7(
    cost_oracle: CostOracle,
    A_seq: Union[Sequence[np.ndarray], Callable[[int], np.ndarray]],
    cfg: Algo7Config,
    proj_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sigma_seq: Optional[Union[Sequence[float], Callable[[int], float]]] = None,
    x0: Optional[np.ndarray] = None,
) -> dict:
    n, d, T = cfg.n, cfg.d, cfg.T
    if proj_X is None:
        proj_X = lambda X: proj_l2_ball(X, radius=1.0)

    rng = np.random.default_rng(cfg.seed)

    if sigma_seq is None:
        sigma_const = compute_sigma_const_from_lemma2(cfg)
        sigma_seq = lambda t: sigma_const

    alpha_t_fn = alpha_schedule_zeta_over_sqrt_t(cfg.zeta)

    w = np.ones((n,), dtype=float)
    z = np.zeros((n, d), dtype=float)

    if x0 is None:
        x = np.zeros((n, d), dtype=float)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != (n, d):
            raise ValueError(f"x0 must have shape {(n, d)}, got {x.shape}")

    x_hist = np.zeros((T + 1, n, d), dtype=float)
    z_hist = np.zeros((T + 1, n, d), dtype=float)
    w_hist = np.zeros((T + 1, n), dtype=float)
    x_hist[0], z_hist[0], w_hist[0] = x, z, w

    for t in range(1, T + 1):
        A = A_seq[t - 1] if isinstance(A_seq, (list, tuple)) else A_seq(t)
        A = np.asarray(A, dtype=float)
        if A.shape != (n, n):
            raise ValueError(f"A_t must have shape {(n, n)}, got {A.shape}")

        # OPGE: u ~ uniform on sphere, query at x + delta u
        u = sample_unit_sphere(n, d, rng)
        x_query = x + cfg.delta * u
        fvals = np.array([cost_oracle(t, i, x_query[i]) for i in range(n)], dtype=float)
        g_tilde = (d / cfg.delta) * fvals[:, None] * u

        # tau noise (Assumption 5) - OPTIONAL
        if cfg.use_tau:
            tau = rng.normal(loc=0.0, scale=float(cfg.tau_std), size=(n, d))
        else:
            tau = np.zeros((n, d), dtype=float)
        g = g_tilde + tau

        # DP noise eta ~ Laplace(0, sigma(t)) elementwise
        sigma_t = sigma_seq[t - 1] if isinstance(sigma_seq, (list, tuple)) else sigma_seq(t)
        eta = rng.laplace(loc=0.0, scale=float(sigma_t), size=(n, d))

        # Push-sum updates
        h = z + eta
        w_new = A @ w
        z_new = (A @ h) + g

        # Prox step with ratio z/w
        alpha_t = alpha_t_fn(t)
        z_over_w = z_new / np.maximum(w_new[:, None], 1e-12)
        x_new = prox_map_quadratic_psi(z_over_w, alpha_t, proj_X)

        w, z, x = w_new, z_new, x_new
        x_hist[t], z_hist[t], w_hist[t] = x, z, w

    return {"x_hist": x_hist, "z_hist": z_hist, "w_hist": w_hist}


# ============================================================
# Corollary 1: optimal delta* (implicit via p_hat(delta))
# ============================================================

def delta_star_corollary1_fixed_point(
    *,
    n: int,
    d: int,
    T: int,
    C: float,
    L: float,
    zeta: float,
    epsilon: float,
    A: np.ndarray,
    l_i: Union[float, np.ndarray] = 0.0,
    delta_init: float = 0.1,
    iters: int = 50,
    tol: float = 1e-10,
) -> float:
    """
    Corollary 1 (correct parsing):
      delta* = sqrt( (4 d C n zeta)/(L c sqrt(T) sqrt(pi*)) * (2 sqrt(2) d p_hat/eps + p_hat) * 1/(1-sqrt(lambda2)) )
    where p_hat = max_i { dC/delta + l_i } (implicit fixed point).
    """
    A = np.asarray(A, dtype=float)
    row_sums = A.sum(axis=1)
    c = float(np.min(row_sums))
    if c <= 0:
        raise ValueError("Invalid A: min row-sum c must be positive.")

    pi = stationary_dist_right_eigvec(A)  # A @ pi = pi, sum(pi)=1
    pi_star = float(np.min(pi))
    if pi_star <= 0:
        raise ValueError("Invalid stationary distribution: pi* must be positive.")

    lam2 = lambda2_magnitude(A)
    lam2 = float(np.clip(lam2, 0.0, 1.0 - 1e-12))
    mix = 1.0 / max(1.0 - np.sqrt(lam2), 1e-12)  # 1/(1 - sqrt(lambda2))

    l = np.asarray(l_i, dtype=float)
    if l.size == 1:
        l = np.full((n,), float(l))

    delta = float(max(delta_init, 1e-12))
    for _ in range(iters):
        p_hat = float(np.max(d * C / max(delta, 1e-12) + l))

        # CORRECT term: (2*sqrt(2)*d*p_hat/epsilon + p_hat)
        term = (2.0 * np.sqrt(2.0) * d * p_hat / epsilon) + p_hat

        numer = 4.0 * d * C * n * zeta * term * mix
        denom = L * c * np.sqrt(T) * np.sqrt(pi_star)
        delta_new = float(np.sqrt(numer / max(denom, 1e-12)))

        if abs(delta_new - delta) <= tol * max(1.0, abs(delta)):
            delta = delta_new
            break

        delta = delta_new

    return float(max(delta, 1e-12))


# ============================================================
# Linear regression on bodyfat dataset
# ============================================================

def _build_pushsum_matrix_from_topology(topology: str, num_nodes: int) -> np.ndarray:
    """Build a column-stochastic push-sum matrix from an undirected topology."""
    base_adj, _, _, _ = _build_network_topology(topology, num_nodes)
    edges: List[Tuple[int, int]] = []
    for i in range(num_nodes):
        for j in np.where(base_adj[i])[0]:
            edges.append((i, int(j)))
    return build_column_stochastic_A(num_nodes, edges, add_self_loops=True)


def run_pbddo_linear_regression_cor1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int = 20000,
    num_nodes: int = 8,
    epsilon: float = 2.0,
    zeta: float = 1.0,
    R: float = 10.0,
    alpha_reg: float = 0.0,
    network_topology: str = "cycle",
    delta_init: Optional[float] = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, dict, float, float, float, float]:
    """
    Apply Algorithm (7) with Corollary 1 parameter choice:
      alpha_t = zeta / sqrt(t)
      delta = delta* (computed by fixed-point iteration)

    Returns:
      w_hat
      out (contains x_hist, z_hist, w_hist)
      mse
      avg_loss_per_round   (averaged over learners, normalized by T)
      avg_regret_per_round (normalized by T)
      delta_star_used
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, dim = X.shape

    # Defines (a_{i,t}, b_{i,t}) for regret computation.
    samples = rng.integers(0, n_samples, size=(T, num_nodes))
    loss_oracle = _make_loss_oracle(X, y, samples, alpha_reg)

    def cost_oracle(t: int, i: int, x: np.ndarray) -> float:
        # t is 1-based here (matching run_algorithm7)
        return loss_oracle(i, t, x)

    # Bound |f_{t,i}(x)| <= C (paper Assumption 4 uses C)
    max_abs_b = float(np.abs(y).max(initial=0.0))
    _, loss_bound = _compute_bounds(R=R, alpha_reg=alpha_reg, max_abs_b=max_abs_b)
    C = float(loss_bound)

    # Lipschitz constant L (paper Assumption 2)
    L = float(lipschitz_bound_squared_loss(X, y, R=R, alpha_reg=alpha_reg))

    # Build push-sum matrix A (fixed topology in this experiment)
    A = _build_pushsum_matrix_from_topology(network_topology, num_nodes)
    A_seq = [A for _ in range(T)]

    # Corollary 1 delta*
    if delta_init is None:
        delta_init = 0.1 * R
    delta_star = delta_star_corollary1_fixed_point(
        n=num_nodes,
        d=dim,
        T=T,
        C=C,
        L=L,
        zeta=zeta,
        epsilon=epsilon,
        A=A,
        l_i=0.0,
        delta_init=float(delta_init),
        iters=50,
    )

    delta_star = min(0.5 * R, delta_star)

    cfg = Algo7Config(
        n=num_nodes,
        d=dim,
        T=T,
        delta=delta_star,
        epsilon=epsilon,
        zeta=zeta,   # alpha_t = zeta/sqrt(t)
        C=C,
        l_i=0.0,
        use_tau=False,
        seed=0,
    )

    projX = lambda X_: proj_l2_ball(X_, radius=R)
    out = run_algorithm7(cost_oracle=cost_oracle, A_seq=A_seq, cfg=cfg, proj_X=projX)

    x_hist = out["x_hist"]

    # Average loss and average regret (both normalized by T)
    avg_loss_per_round, avg_regret_per_round = compute_avg_loss_and_avg_regret_per_round(
        X_hist=x_hist,
        samples=samples,
        X=X,
        y=y,
        alpha_reg=alpha_reg,
        R=R,
        use_pre_update_iterate=True,  # uses x_hist[t-1] for round t
    )

    # Global model: average across nodes and time
    w_hat = x_hist.mean(axis=(0, 1))
    preds = X @ w_hat
    mse = float(np.mean((preds - y) ** 2))

    return w_hat, out, mse, avg_loss_per_round, avg_regret_per_round, float(delta_star)


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
    zeta = 1.0

    for alpha_reg in (0.0,):
        w_hat, out, mse, avg_loss, avg_regret, delta_star = run_pbddo_linear_regression_cor1(
            X,
            y,
            T=T,
            num_nodes=num_nodes,
            epsilon=epsilon,
            zeta=zeta,
            R=R,
            alpha_reg=alpha_reg,
            network_topology=network_topology,
            delta_init=0.1 * R,
            rng=np.random.default_rng(42),
        )

        print("=" * 60)
        print(f"alpha_reg = {alpha_reg}")
        print(f"delta_star (Corollary 1) = {delta_star:.6g}")
        print(f"Final model shape: {w_hat.shape}")
        print(f"Mean squared error: {mse:.6f}")
        print(f"x_hist shape: {out['x_hist'].shape}")
        print(f"Avg loss per round (normalized by T):  {avg_loss:.6f}")
        print(f"Avg regret per round (normalized by T): {avg_regret:.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
