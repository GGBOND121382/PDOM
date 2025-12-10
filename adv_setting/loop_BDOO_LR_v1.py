from sklearn.datasets import load_diabetes
import numpy as np

from optimization_utils.BanditLogisticRegression import run_bandit_logistic_regression


# assume run_bandit_logistic_regression is defined in this file


def example_logistic_diabetes():
    data = load_diabetes()
    X = data.data               # (n_samples, d)
    y_reg = data.target         # continuous

    # Binary labels: above-median vs below-median
    y = (y_reg > np.median(y_reg)).astype(int)

    T = 2000
    num_nodes = 5
    R = 5.0
    r = 1.0
    rho = 0.5

    # 1) No regularization (convex setting)
    w_no, X_hist_no, losses_no, acc_no = run_bandit_logistic_regression(
        X=X,
        y=y,
        T=T,
        num_nodes=num_nodes,
        R=R,
        r=r,
        rho=rho,
        alpha_reg=0.0,      # no L2
    )

    # 2) With L2 regularization term 0.5 * alpha_reg * ||w||^2
    w_reg, X_hist_reg, losses_reg, acc_reg = run_bandit_logistic_regression(
        X=X,
        y=y,
        T=T,
        num_nodes=num_nodes,
        R=R,
        r=r,
        rho=rho,
        alpha_reg=1.0,      # > 0 => strongly_convex schedule
    )

    print("Training accuracy (no reg) :", acc_no)
    print("Training accuracy (with reg):", acc_reg)


if __name__ == '__main__':
    example_logistic_diabetes()
