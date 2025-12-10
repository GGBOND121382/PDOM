import numpy as np
import math
from sklearn.preprocessing import MaxAbsScaler


# assume run_algorithm2_bandit_paper_params is defined above in the same file


def _prepare_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary targets to labels in {-1, +1}.
      - If y in {0,1}, map to {-1,+1};
      - If y in {-1,+1}, keep as is;
      - Else, threshold at median.
    """
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if np.array_equal(uniq, np.array([0, 1])):
        return 2 * y - 1
    if np.array_equal(uniq, np.array([-1, 1])):
        return y
    thresh = np.median(y)
    return np.where(y >= thresh, 1.0, -1.0)


def _split_dataset_round_robin(
    X: np.ndarray,
    y_pm: np.ndarray,
    num_nodes: int,
):
    """
    Split dataset across nodes in a round-robin fashion.
    Returns X_list, y_list (each length = num_nodes).
    """
    n, d = X.shape
    X_list = [[] for _ in range(num_nodes)]
    y_list = [[] for _ in range(num_nodes)]
    for idx in range(n):
        node = idx % num_nodes
        X_list[node].append(X[idx])
        y_list[node].append(y_pm[idx])
    X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]
    y_list = [np.asarray(yi, dtype=float) for yi in y_list]
    return X_list, y_list


def _compute_logistic_constants(
    X_list,
    alpha_reg: float,
    R: float,
):
    """
    Compute Lipschitz constant L_f and uniform bound C for
    f(w) = logistic + 0.5 * alpha_reg * ||w||^2 over ||w|| <= R.
    """
    max_norm_a = 0.0
    for Xi in X_list:
        if Xi.size == 0:
            continue
        norms = np.linalg.norm(Xi, axis=1)
        max_norm_a = max(max_norm_a, float(norms.max(initial=0.0)))

    # Gradient bound: ||∇ℓ_i(w)|| ≤ ||a_i||, reg grad = alpha_reg * w
    # so ||∇f_i(w)|| ≤ max_norm_a + alpha_reg * R
    L_f = max_norm_a + alpha_reg * R

    # Function value bound:
    # logistic term ≤ log(1 + exp(||a|| * R)), ||a|| ≤ max_norm_a
    logistic_max = math.log(1.0 + math.exp(max_norm_a * R))
    reg_max = 0.5 * alpha_reg * (R ** 2)
    C = logistic_max + reg_max
    return L_f, C


def make_logistic_loss_oracle(
    X_list,
    y_list,
    alpha_reg: float = 0.0,
):
    """
    Build loss_oracle(i,t,w) for bandit logistic regression:

      f_{i,t}(w) = log(1 + exp(-y a^T w))
                   + 0.5 * alpha_reg * ||w||^2

    At time t, node i uses its local sample indexed by (t-1) % n_i.
    """
    num_nodes = len(X_list)
    sizes = [Xi.shape[0] for Xi in X_list]

    def loss_oracle(i: int, t: int, w: np.ndarray) -> float:
        Xi = X_list[i]
        yi = y_list[i]
        n_i = sizes[i]
        if n_i == 0:
            # just pure regularization if this node has no data
            return 0.5 * alpha_reg * float(np.dot(w, w))

        idx = (t - 1) % n_i
        a = Xi[idx]
        y = yi[idx]  # in {-1, +1}
        z = y * float(np.dot(a, w))

        # numerically stable logistic loss
        if z > 0:
            logistic = math.log(1.0 + math.exp(-z))
        else:
            logistic = -z + math.log(1.0 + math.exp(z))

        reg = 0.5 * alpha_reg * float(np.dot(w, w))
        return logistic + reg

    return loss_oracle


def run_bandit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    num_nodes: int,
    R: float,
    r: float,
    rho: float,
    alpha_reg: float = 0.0,
    p_edge: float = 0.5,
    a: float = None,
    setting: str = None,
    rng: np.random.Generator = None,
):
    """
    Apply Algorithm 2 to a logistic regression task (with optional L2).

    Parameters
    ----------
    X : (n_samples, d)
    y : targets (any real; will be converted to {-1, +1})
    T : time horizon (rounds of Algorithm 2)
    num_nodes : number of nodes N
    R, r : geometry of K: rB ⊆ K ⊆ RB
    rho : network connectivity parameter for Algorithm 2
    alpha_reg : L2 coefficient in 0.5 * alpha_reg * ||w||^2
                (set 0 to remove regularization)
    p_edge : ER edge probability
    a : consensus weight (defaults to 1/N inside Algorithm 2 if None)
    setting : "convex" or "strongly_convex"; if None:
              - "convex"  if alpha_reg == 0
              - "strongly_convex" if alpha_reg > 0
    rng : np.random.Generator

    Returns
    -------
    w_hat : np.ndarray, shape (d,)
        Final global model = average over time and nodes.
    X_hist : np.ndarray, shape (T+1, N, d)
        Iterates from Algorithm 2.
    losses : np.ndarray, shape (T, N)
        Observed bandit losses.
    acc : float
        Training accuracy of w_hat on (X, y).
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) scale features (keeps norms under control)
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    d = X_scaled.shape[1]

    # 2) prepare labels in {-1,+1}, split across nodes
    y_pm = _prepare_binary_labels(y)
    X_list, y_list = _split_dataset_round_robin(X_scaled, y_pm, num_nodes)

    # 3) loss oracle
    loss_oracle = make_logistic_loss_oracle(X_list, y_list, alpha_reg=alpha_reg)

    # 4) Lipschitz constant and loss bound
    L_f, C = _compute_logistic_constants(X_list, alpha_reg=alpha_reg, R=R)

    # 5) choose Algorithm 2 "setting" and strong convexity parameter
    if setting is None:
        setting = "strongly_convex" if alpha_reg > 0 else "convex"

    if setting == "strongly_convex":
        if alpha_reg <= 0:
            raise ValueError("alpha_reg must be > 0 for strongly_convex setting.")
        alpha_sc = alpha_reg  # f(w) is alpha_reg-strongly convex
    else:
        alpha_sc = None

    # 6) run your Algorithm 2
    X_hist, losses = run_algorithm2_bandit_paper_params(
        loss_oracle=loss_oracle,
        T=T,
        num_nodes=num_nodes,
        dim=d,
        R=R,
        r=r,
        L_f=L_f,
        C=C,
        rho=rho,
        alpha=alpha_sc,
        setting=setting,
        a=a,
        p_edge=p_edge,
        rng=rng,
    )

    # 7) global model = average over time and nodes
    W_avg_nodes = X_hist.mean(axis=0)   # (N, d)
    w_hat = W_avg_nodes.mean(axis=0)    # (d,)

    # 8) training accuracy of w_hat
    logits = X_scaled @ w_hat
    y_pred_pm = np.where(logits >= 0.0, 1.0, -1.0)
    y_true_pm = _prepare_binary_labels(y)
    acc = float((y_pred_pm == y_true_pm).mean())

    return w_hat, X_hist, losses, acc



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
