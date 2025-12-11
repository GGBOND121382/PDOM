from sklearn.datasets import load_diabetes
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import torch

from optimization_utils.BanditLogisticRegression import (
    _build_cycle_graph,
    _build_gossip_matrix,
    _compute_max_degree,
    run_algorithm2_bandit_paper_params,
)
from optimization_utils.BanditLogisticRegression_v1 import _prepare_binary_labels, _split_dataset_round_robin, \
    make_logistic_loss_oracle, _compute_logistic_constants
from optimization_utils.config_save_load import conf_load


# assume run_bandit_logistic_regression is defined in this file


def run_bandit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    num_nodes: int,
    R: float,
    r: float,
    rho: float = None,
    alpha_reg: float = 0.0,
    p_edge: float = 0.5,
    a: float = None,
    setting: str = None,
    graph_mode: str = "cycle",   # NEW: default = cycle = BDOO experiments
    rng: np.random.Generator = None,
):
    """
    Apply BDOO (Algorithm 2 with one-point bandit feedback) to a
    (regularized) logistic regression task.

    Parameters
    ----------
    X : (n_samples, d)
        Preprocessed features (can be raw; we rescale inside).
    y : targets (any real; will be converted to {-1, +1}).
    T : time horizon (rounds of Algorithm 2).
    num_nodes : number of nodes N.
    R, r : geometry of K: rB ⊆ K ⊆ RB.
    rho : network connectivity parameter for BDOO. If None, set to the
        second-largest eigenvalue of the gossip matrix induced by the
        cycle graph and consensus weight ``a``.
    alpha_reg : L2 coefficient in 0.5 * alpha_reg * ||w||^2
                (set 0 to remove regularization).
    p_edge : ER edge probability (used only if graph_mode="er").
    a : consensus weight (defaults to 1 / (1 + max_degree) for the cycle graph
        if None).
    setting : "convex" or "strongly_convex"; if None:
              - "convex"  if alpha_reg == 0
              - "strongly_convex" if alpha_reg > 0
    graph_mode : {"cycle", "er"}
        Communication graph mode passed to BDOO.
    rng : np.random.Generator

    Returns
    -------
    w_hat : np.ndarray, shape (d,)
        Final global model = average over time and nodes.
    X_hist : np.ndarray, shape (T+1, N, d)
        Iterates from BDOO (Algorithm 2).
    losses : np.ndarray, shape (T, N)
        Observed bandit losses.
    acc : float
        Training accuracy of w_hat on (X, y).
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) scale features:
    #    - per-feature to [-1, 1] via MaxAbsScaler
    #    - per-sample to unit L2 norm, as in the implementation details
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X.astype(float))
    row_norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    X_scaled = X_scaled / row_norms
    d = X_scaled.shape[1]

    # 2) prepare labels in {-1,+1}, split across nodes (round-robin here;
    #    you can replace this by your "positive to small i, negative to
    #    large i + label flipping" logic if you want to exactly match the paper).
    y_pm = _prepare_binary_labels(y)
    X_list, y_list = _split_dataset_round_robin(X_scaled, y_pm, num_nodes)

    # 3) loss oracle
    loss_oracle = make_logistic_loss_oracle(X_list, y_list, alpha_reg=alpha_reg)

    # 4) Lipschitz constant and loss bound
    L_f, C = _compute_logistic_constants(X_list, alpha_reg=alpha_reg, R=R)

    print("Lipschitz constant: ", L_f)
    print("loss bound:", C)


    # 5) choose BDOO "setting" and strong convexity parameter
    if setting is None:
        setting = "strongly_convex" if alpha_reg > 0 else "convex"

    if setting == "strongly_convex":
        if alpha_reg <= 0:
            raise ValueError("alpha_reg must be > 0 for strongly_convex setting.")
        alpha_sc = alpha_reg  # f(w) is alpha_reg-strongly convex
    else:
        alpha_sc = None

    if graph_mode != "cycle":
        raise ValueError("rho computation currently supports only the cycle graph mode.")

    base_adj = _build_cycle_graph(num_nodes)
    if a is None:
        a = 1.0 / (1.0 + _compute_max_degree(base_adj))

    gossip_matrix = _build_gossip_matrix(base_adj, a)
    eigenvalues = np.linalg.eigvals(gossip_matrix)
    eigenvalues_sorted = np.sort(np.real(eigenvalues))[::-1]
    rho_value = float(eigenvalues_sorted[1])

    # 6) run BDOO (Algorithm 2)
    X_hist, losses = run_algorithm2_bandit_paper_params(
        loss_oracle=loss_oracle,
        T=T,
        num_nodes=num_nodes,
        dim=d,
        R=R,
        r=r,
        L_f=L_f,
        C=C,
        rho=rho_value if rho is None else rho,
        alpha=alpha_sc,
        setting=setting,
        a=a,
        p_edge=p_edge,
        graph_mode=graph_mode,
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



def load_dataset_from_conf():
    """Load and preprocess dataset based on ``adv_setting/conf.ini`` settings."""

    conf_dict = conf_load()
    data = conf_dict["data"]
    target = conf_dict["target"]
    is_minus_one = conf_dict["is_minus_one"]

    X = torch.load(data, weights_only=False)
    X = MaxAbsScaler().fit_transform(X)
    X = preprocessing.normalize(X, norm="l2")

    y = torch.load(target, weights_only=False)
    y = np.array(y, dtype=int)
    if is_minus_one:
        # Transform labels from {-1, 1} to {0, 1} when needed.
        y = (y + 1) / 2
    y = np.array(y, dtype=int).reshape(-1)

    return conf_dict, X, y


if __name__ == '__main__':
    conf_dict, X_loaded, y_loaded = load_dataset_from_conf()

    print(y_loaded.shape)

    R = 10.0  # feasible region radius
    r = R  # inner-ball radius (choose any 0 < r < R)
    alpha_reg = 1e-3  # example; you’ll sweep {1e-1, ..., 1e-5}
    T = 20000  # for KDDCup99, or 12500 for Diabetes

    # w_hat, X_hist, losses, acc = run_bandit_logistic_regression(
    #     X_loaded,
    #     y_loaded,
    #     T=T,
    #     num_nodes=conf_dict.get("number_of_clients", 8),
    #     R=R,
    #     r=r,
    #     alpha_reg=alpha_reg,
    #     graph_mode="cycle",  # <<< BDOO: cycle network
    #     rng=np.random.default_rng(42),
    # )

    w_hat, X_hist, losses, acc = run_bandit_logistic_regression(
        X_loaded,
        y_loaded,
        T=T,
        num_nodes=conf_dict.get("number_of_clients", 8),
        R=R,
        r=r,
        alpha_reg=0.,
        graph_mode="cycle",  # <<< BDOO: cycle network
        rng=np.random.default_rng(42),
    )

