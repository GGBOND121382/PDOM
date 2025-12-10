from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from optimization_utils.BanditLogisticRegression import run_algorithm2_bandit_paper_params
from optimization_utils.BanditLogisticRegression_v1 import _prepare_binary_labels, _split_dataset_round_robin, \
    make_logistic_loss_oracle, _compute_logistic_constants


# assume run_bandit_logistic_regression is defined in this file


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
    rho : network connectivity parameter for BDOO.
    alpha_reg : L2 coefficient in 0.5 * alpha_reg * ||w||^2
                (set 0 to remove regularization).
    p_edge : ER edge probability (used only if graph_mode="er").
    a : consensus weight (defaults to 1/N inside BDOO if None).
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

    # 5) choose BDOO "setting" and strong convexity parameter
    if setting is None:
        setting = "strongly_convex" if alpha_reg > 0 else "convex"

    if setting == "strongly_convex":
        if alpha_reg <= 0:
            raise ValueError("alpha_reg must be > 0 for strongly_convex setting.")
        alpha_sc = alpha_reg  # f(w) is alpha_reg-strongly convex
    else:
        alpha_sc = None

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
        rho=rho,
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



if __name__ == '__main__':
    R = 10.0  # feasible region radius
    r = R  # inner-ball radius (choose any 0 < r < R)
    rho = 0.5  # or whatever you use from theory / tuning
    alpha_reg = 1e-3  # example; you’ll sweep {1e-1, ..., 1e-5}
    T = 20000  # for KDDCup99, or 12500 for Diabetes

    w_hat, X_hist, losses, acc = run_bandit_logistic_regression(
        X_preprocessed,  # your dataset matrix
        y_labels,  # raw labels
        T=T,
        num_nodes=8,
        R=R,
        r=r,
        rho=rho,
        alpha_reg=alpha_reg,
        graph_mode="cycle",  # <<< BDOO: cycle network
        rng=np.random.default_rng(42),
    )

