import numpy as np
import math
from typing import Callable, Optional, Tuple


def _project_to_l2_ball(X: np.ndarray, R: float) -> np.ndarray:
    """
    Project each row of X onto the ℓ2 ball of radius R.
    X: (N, d)
    """
    if R <= 0:
        return np.zeros_like(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    scale = np.minimum(1.0, R / norms)
    return X * scale


def _sample_unit_vectors(rng: np.random.Generator, shape) -> np.ndarray:
    """
    Sample unit vectors uniformly on the sphere S^{d-1}.
    shape: (N, d)
    """
    Z = rng.normal(size=shape)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return Z / norms


def _build_er_gossip_matrix(
    num_nodes: int,
    p_edge: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a simple symmetric ER graph and the associated row-stochastic
    gossip matrix P (Metropolis-like but simplified).

    P is (N, N), row-stochastic, with P[i, j] > 0 only if (i, j) is an edge
    (including self-loops).
    """
    N = num_nodes

    if N <= 1:
        return np.eye(N, dtype=float)

    # Undirected ER graph
    if p_edge <= 0.0:
        # fallback: cycle graph
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            adj[i, i] = True
            adj[i, (i - 1) % N] = True
            adj[i, (i + 1) % N] = True
    else:
        upper = rng.random((N, N))
        adj = (upper < p_edge)
        adj = np.triu(adj, k=1)
        adj = adj + adj.T
        np.fill_diagonal(adj, True)

    deg = adj.sum(axis=1)
    P = np.zeros((N, N), dtype=float)

    # Simple row-stochastic weights: 1/deg(i) on neighbors (incl. self)
    for i in range(N):
        if deg[i] == 0:
            P[i, i] = 1.0
        else:
            P[i, adj[i]] = 1.0 / float(deg[i])

    return P


def _build_gossip_matrix(adj: np.ndarray, a: float) -> np.ndarray:
    """
    Build the row-stochastic gossip matrix matching
    mixed = (1 - a * deg_i) * Y[i] + a * sum_{j in N(i)} Y[j].

    Parameters
    ----------
    adj : np.ndarray of bool
        Symmetric adjacency matrix without self-loops.
    a : float
        Consensus weight.
    """
    num_nodes = adj.shape[0]
    P = np.zeros((num_nodes, num_nodes), dtype=float)
    degrees = adj.sum(axis=1)

    for i in range(num_nodes):
        neighbors = np.where(adj[i])[0]
        deg_i = len(neighbors)

        if deg_i == 0:
            P[i, i] = 1.0
            continue

        P[i, i] = 1.0 - a * deg_i
        for j in neighbors:
            P[i, j] += a

    return P


def _compute_max_degree(adj: np.ndarray) -> float:
    """Return the maximum degree of the (boolean) adjacency matrix."""
    if adj.size == 0:
        return 0.0
    return float(adj.sum(axis=1).max(initial=0))


def _build_cycle_graph(num_nodes: int) -> np.ndarray:
    """
    Build an undirected cycle graph on {0, ..., num_nodes-1}.
    Each node has exactly two neighbors (except num_nodes <= 2).
    """
    if num_nodes <= 1:
        return np.zeros((num_nodes, num_nodes), dtype=bool)

    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
    for i in range(num_nodes):
        j_next = (i + 1) % num_nodes
        adj[i, j_next] = True
        adj[j_next, i] = True
    return adj


def run_algorithm2_bandit_paper_params(
    loss_oracle: Callable[[int, int, np.ndarray], float],
    T: int,
    num_nodes: int,
    dim: int,
    # Geometry of K:  rB ⊆ K ⊆ RB
    R: float,
    r: float,
    # Lipschitz / boundedness constants (Assumption 5 + (8)):
    L_f: float,
    C: float,
    # Network connectivity parameter ρ (from paper, depends on a, graph):
    rho: float,
    # Strong convexity parameter α if using "strongly_convex" setting
    alpha: Optional[float] = None,
    setting: str = "convex",   # "convex" or "strongly_convex"
    # Consensus weight a (must satisfy 0 < a ≤ 1 / (1 + max_i |N_i|)):
    a: Optional[float] = None,
    # Probability for Erdős–Rényi graph G_t (only used if graph_mode == "er"):
    p_edge: float = 0.5,
    # NEW: graph mode for BDOO experiments
    graph_mode: str = "cycle",   # "cycle" (default BDOO) or "er"
    # Optionally provide a fixed adjacency to override graph_mode
    base_adj: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BDOO baseline (YanSVQ13-style, in our bandit setting).

    One-point bandit DOO with:
      - Projection onto ℓ2-ball K = {w : ||w||_2 ≤ R}
      - Consensus over a communication graph
      - Stepsizes / smoothing as in our Algorithm 2 (Theorems 3, 4).

    Parameters
    ----------
    loss_oracle : callable
        f(i, t, x) → scalar loss f_{i,t}(x).
    T : int
        Time horizon.
    num_nodes : int
        Number of nodes N.
    dim : int
        Dimension d.
    R, r : float
        K satisfies rB ⊆ K ⊆ RB.
    L_f : float
        Lipschitz constant for losses over K.
    C : float
        Uniform bound on |f_{i,t}(x)| over i, t, x ∈ K.
    rho : float
        Connectivity parameter ρ (you can compute it for the cycle or
        just treat it as a tuning constant).
    alpha : float, optional
        Strong convexity parameter α (for strongly_convex setting).
    setting : {"convex", "strongly_convex"}
        Which case to use: convex or strongly_convex.
    a : float, optional
        Consensus weight. If None, defaults to 1 / (1 + max_degree) of the
        fixed communication graph.
    p_edge : float
        Used only if graph_mode == "er".
    graph_mode : {"cycle", "er"}
        - "cycle" (default): fixed N-node cycle graph  (BDOO experiments).
        - "er": time-varying Erdős–Rényi graphs as in the original code.
    base_adj : np.ndarray, optional
        If not None, must be a (num_nodes, num_nodes) boolean adjacency
        matrix. Overrides graph_mode and is used at every round.
    rng : np.random.Generator, optional
        RNG; default np.random.default_rng().

    Returns
    -------
    X_hist : np.ndarray
        Shape (T + 1, num_nodes, dim), X_hist[t, i] = x_{i,t}.
    losses : np.ndarray
        Shape (T, num_nodes), losses[t-1, i] = f_{i,t}(x_{i,t} + δ u_{i,t}).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ====== Hyperparameters (same as before) ======
    N = num_nodes
    d = dim

    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1).")

    # Convex case (Theorem 3 in the paper):
    #   η_t = 2 δ R / (d C sqrt(t))
    #   δ = (c1 / c2)^{1/2} T^{-1/4}
    #   ξ = δ / r
    #   c1 = 3 d R C (1 + 4 ρ (1 + sqrt(N)) / (1 - ρ))
    #   c2 = 2 (L_f + C / r)
    if setting == "convex":
        c1 = 3.0 * d * R * C * (1.0 + 4.0 * rho * (1.0 + np.sqrt(N)) / (1.0 - rho))
        c2 = 2.0 * (L_f + C / r)
        print("c1:", c1)
        print("c2:", c2)
        delta = np.sqrt(c1 / c2) * (T ** (-0.25))
        xi = delta / r

        def eta_t(t: int) -> float:
            return 2.0 * delta * R / (d * C * np.sqrt(float(t)))

    # Strongly convex case (Theorem 4 in the paper):
    #   η_t = 1 / (α t)
    #   δ = (2 c3 (1 + ln(T)) / (c2 T))^{1/3}
    #   ξ = δ / r
    #   c3 = d C^2 / (2 α) (1 + 6 ρ (1 + sqrt(N)) / (1 - ρ))
    #   c2 = 2 (L_f + C / r)
    elif setting == "strongly_convex":
        if alpha is None or alpha <= 0:
            raise ValueError("alpha > 0 is required for 'strongly_convex' setting.")

        c3 = (d * (C ** 2) / (2.0 * alpha)) * (
            1.0 + 6.0 * rho * (1.0 + np.sqrt(N)) / (1.0 - rho)
        )
        c2 = 2.0 * (L_f + C / r)
        delta = ((2.0 * c3 * (1.0 + np.log(T))) / (c2 * T)) ** (1.0 / 3.0)
        xi = delta / r

        def eta_t(t: int) -> float:
            return 1.0 / (alpha * float(t))

    else:
        raise ValueError("setting must be 'convex' or 'strongly_convex'.")

    print("xi: ", xi)
    if xi >= 1.0:
        raise ValueError("xi = delta / r must be < 1; check R, r, L_f, C, rho, T.")

    # ====== Adjacency for BDOO ======
    # If base_adj is given, we always use it.
    # Else, for BDOO experiments: static cycle graph ("cycle" mode).
    # If you want time-varying ER graphs, set graph_mode="er".
    if base_adj is not None:
        if base_adj.shape != (num_nodes, num_nodes):
            raise ValueError("base_adj must have shape (num_nodes, num_nodes).")
        adj_fixed = base_adj.astype(bool)
    elif graph_mode == "cycle":
        adj_fixed = _build_cycle_graph(num_nodes)
    else:
        adj_fixed = None  # will sample ER each round

    if a is None:
        if adj_fixed is None:
            raise ValueError(
                "Default consensus weight requires a fixed adjacency; "
                "please supply 'a' explicitly."
            )
        max_degree = _compute_max_degree(adj_fixed)
        a = 1.0 / (1.0 + max_degree)

    gossip_fixed = None
    if adj_fixed is not None:
        gossip_fixed = _build_gossip_matrix(adj_fixed, a)

    # ====== Main algorithm ======
    X_hist = np.zeros((T + 1, num_nodes, dim), dtype=float)  # x_{i,1} = 0
    losses = np.zeros((T, num_nodes), dtype=float)

    # Projection radius for (1 - ξ)K
    R_shrunk = (1.0 - xi) * R

    for t in range(1, T + 1):
        x_t = X_hist[t - 1]  # shape (num_nodes, dim)

        # 1) sample directions u_{i,t} on unit sphere
        U = np.zeros_like(x_t)
        for i in range(num_nodes):
            U[i] = _sample_unit_sphere(dim, rng)

        # 2) observe one-point bandit feedback f_{i,t}(x_{i,t} + δ u_{i,t})
        f_vals = np.zeros(num_nodes, dtype=float)
        for i in range(num_nodes):
            query_point = x_t[i] + delta * U[i]
            f_vals[i] = loss_oracle(i, t, query_point)
        losses[t - 1] = f_vals

        # 3) gradient estimators:
        #    g_{i,t} = (d / δ) f_{i,t}(x_{i,t} + δ u_{i,t}) u_{i,t}
        coeff = d / delta
        G = coeff * f_vals[:, None] * U  # shape (num_nodes, dim)

        # 4) local gradient step y_{i,t} = x_{i,t} - η_t g_{i,t}
        eta = eta_t(t)
        Y = x_t - eta * G

        # 5) communication graph
        if adj_fixed is not None:
            adj = adj_fixed
            mixed_all = gossip_fixed @ Y
        else:
            raise ValueError(
                "Time-varying ER graphs are not currently supported in this implementation."
            )

        # 6) consensus + projection onto (1 - ξ)K
        X_next = np.zeros_like(x_t)
        for i in range(num_nodes):
            mixed = mixed_all[i]
            X_next[i] = _proj_l2_ball(mixed, R_shrunk)

        X_hist[t] = X_next

    return X_hist, losses






# ==============================
# Helpers
# ==============================

def _proj_l2_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Euclidean projection of x onto the L2 ball of radius `radius` centered at 0.
    """
    norm = np.linalg.norm(x)
    if norm <= radius or radius <= 0:
        return x
    return x * (radius / norm)


def _sample_unit_sphere(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a point uniformly on the unit sphere S^{dim-1}.
    """
    v = rng.normal(size=dim)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v[0] = 1.0
        norm = 1.0
    return v / norm
