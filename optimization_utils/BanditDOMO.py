import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import load_diabetes
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import torch
import random
import math
import copy
import sys
sys.path.append('../')
from optimization_utils.LogisticRegression import *

import os

print(os.getcwd())


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# L2正则化
class l2_regularization():
    def __init__(self, alpha=1.):
        self.alpha = alpha

    # L2正则化的方差
    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)

    # L2正则化的梯度
    def grad(self, w):
        return self.alpha * w


import numpy as np
from typing import Callable, Optional, Tuple


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


def _sample_er_graph(num_nodes: int,
                     p_edge: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Sample an undirected Erdős–Rényi graph G(n, p_edge) without self-loops.

    Returns
    -------
    adj : np.ndarray, shape (num_nodes, num_nodes), dtype=bool
        Symmetric adjacency matrix with zeros on the diagonal.
    """
    if num_nodes <= 1:
        return np.zeros((num_nodes, num_nodes), dtype=bool)

    upper = rng.random(size=(num_nodes, num_nodes))
    mask = np.triu(np.ones_like(upper, dtype=bool), k=1)
    edges = (upper < p_edge) & mask

    adj = edges | edges.T
    np.fill_diagonal(adj, False)
    return adj


# ==============================
# Algorithm 2 with paper hyperparameters
# ==============================

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
    # Network connectivity parameter ρ (from paper, depends on a, p, G):
    rho: float,
    # Strong convexity parameter α if using "strongly_convex" setting
    alpha: Optional[float] = None,
    setting: str = "convex",   # "convex" or "strongly_convex"
    # Consensus weight a (must satisfy 0 < a ≤ 1 / (1 + max_i |N_i|)):
    a: Optional[float] = None,
    # Probability for Erdős–Rényi graph G_t:
    p_edge: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distributed online one-point bandit algorithm (Algorithm 2) with the
    SAME hyperparameter choices as in the paper (Theorems 3 and 4). :contentReference[oaicite:1]{index=1}

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
        K satisfies rB ⊆ K ⊆ RB (Assumption 4).
    L_f : float
        Lipschitz constant for losses over K (Assumption 5).
    C : float
        Uniform bound on |f_{i,t}(x)| over i, t, x ∈ K (equation (8)).
    rho : float
        ρ from the paper (algebraic connectivity of the expected network).
    alpha : float, optional
        Strong convexity parameter α (Assumption 3).
        Required if setting == "strongly_convex".
    setting : {"convex", "strongly_convex"}
        Which theorem to follow:
        - "convex" → Theorem 3
        - "strongly_convex" → Theorem 4
    a : float, optional
        Consensus weight. If None, defaults to 1 / num_nodes (valid for
        complete base graph with max degree N-1).
    p_edge : float
        Link probability for Erdős–Rényi graph at each step.
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

    if a is None:
        # For base graph = complete graph, max_i |N_i| = N-1, so 1/(1+max) = 1/N.
        a = 1.0 / float(num_nodes)

    # ====== Hyperparameters from the paper ======
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

        # 5) sample random ER graph G_t
        adj = _sample_er_graph(num_nodes, p_edge, rng)

        # 6) consensus + projection onto (1 - ξ)K
        X_next = np.zeros_like(x_t)
        for i in range(num_nodes):
            neighbors = np.where(adj[i])[0]
            deg_i = len(neighbors)

            if deg_i == 0:
                mixed = Y[i]
            else:
                mixed = (1.0 - a * deg_i) * Y[i] + a * Y[neighbors].sum(axis=0)

            X_next[i] = _proj_l2_ball(mixed, R_shrunk)

        X_hist[t] = X_next

    return X_hist, losses
