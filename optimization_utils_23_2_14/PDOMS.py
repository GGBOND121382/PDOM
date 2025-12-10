import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# from sklearn.datasets import load_boston, load_diabetes
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import torch
import random
import math
import copy
import sys
sys.path.append('../')
from optimization_utils.LogisticRegression import *

'''
def nine_cycle():
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(9):
        graph.append((i, (i + 1) % 9))
        graph.append((i, (i - 1) % 9))
    #print(graph)
    graph_mat = np.eye(9)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 3.
    return graph_mat

def two2seven_cycle():
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(128):
        graph.append((i, (i + 1) % 128))
        graph.append((i, (i - 1) % 128))
    #print(graph)
    graph_mat = np.eye(128)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 3.
    return graph_mat
'''


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


class PDOMS():
    """
    Parameters:
    -----------
    n_iterations: int
        梯度下降的轮数
    learning_rate: float
        梯度下降学习率
    regularization: l1_regularization or l2_regularization or None
        正则化
    gradient: Bool
        是否采用梯度下降法或正规方程法。
        若使用了正则化，暂只支持梯度下降
    """

    def __init__(self, learning_rate=0.0001, regularization=l2_regularization, gradient=True, eps=0.001,
                 num_client=8, alpha=0., is_private=True, is_lagged=True, radius=10.):
        # self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.eps = eps
        self.num_client = num_client
        self.alpha = alpha
        self.temp_of_data = 0
        self.is_private = is_private
        self.is_lagged = is_lagged
        self.radius = radius
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization(alpha=self.alpha)

    def initialize_weights(self, n_features):
        # 初始化参数
        # limit = np.sqrt(1 / n_features)
        # self.w = np.random.uniform(0, 0, (n_features + 1, 1))
        self.w = np.random.uniform(-self.radius, self.radius, (n_features + 1, 1))
        self.w = projection(self.w, radius=self.radius, is_l2_norm=True)
        # w = np.random.uniform(0, 0, (n_features, 1))

    def fit(self, X, y, mu):
        m_samples, n_features = X.shape
        num_dataofclient = int(m_samples / self.num_client)
        self.initialize_weights(n_features)
        # print(X.shape)
        X = np.insert(X, 0, 1, axis=1)
        # print(X.shape)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = []
        self.class_errors = []
        w_temp = self.w
        tt = 0
        L_G = 1. / 2 + self.alpha
        # L_cons = math.sqrt(2) + self.alpha * math.sqrt(n_features + 1)
        L_cons = math.sqrt(2) + self.alpha * self.radius
        d = n_features + 1
        # C_norm = 2 * math.sqrt(d)
        C_norm = 2 * self.radius

        if (self.is_private and self.alpha > 0):
            tau = 2 * mu + int(d / (self.num_client * self.eps))
            S = 2 * L_cons * math.sqrt(d)
            sigma = S / self.eps
        elif (self.is_private and self.alpha == 0):
            tau = 2 * mu + int(d ** (2. / 3) * num_dataofclient ** (1. / 3) / (
                    (self.num_client * self.eps) ** (2. / 3)))
            S = 2 * L_cons * math.sqrt(d)
            sigma = S / self.eps
        elif (not self.is_private and self.alpha == 0):
            tau = mu + math.ceil((mu * num_dataofclient / self.num_client) ** (1. / 3))
            sigma = 0
        else:
            assert False
        sigma_bar = math.sqrt(L_cons ** 2 + 2 * d * sigma ** 2 / (self.num_client * (tau - mu)))
        print("tau:", tau)
        batch_cnt = 1
        inner_cnt = 1
        if self.gradient == True:
            # 梯度下降
            while True:
                self.temp_of_data = tt
                if self.temp_of_data == num_dataofclient:
                    break
                num1 = self.num_client * (self.temp_of_data)
                num2 = self.num_client * (self.temp_of_data + 1)
                for i in range(num1, num2):
                    # S = 2 * L_cons / (self.alpha * batch_cnt * tau)
                    if (self.is_private and self.alpha > 0):
                        # sigma = math.sqrt(2*(S**2)*math.log(1.25/self.delta)/self.eps**2)
                        learning_rate = 1 / (self.alpha * self.num_client * batch_cnt * (tau - mu))
                    elif (self.alpha == 0):
                        # sigma = 0.
                        tmp = L_G + sigma_bar * math.sqrt(2 * batch_cnt) / (
                                math.sqrt(self.num_client * (tau - mu)) * C_norm)
                        tmp *= self.num_client * (tau - mu)
                        learning_rate = 1 / tmp

                    x_train = np.reshape(X[i], (1, len(X[i])))
                    y_train = y[i]

                    # the update procedure
                    h_x = x_train.dot(w_temp)
                    h_x = np.array(h_x)
                    # print("h_x shape:", h_x.shape)
                    y_pred = sigmoid(h_x[0, 0])
                    # loss = np.mean(y_train * np.log(1+ np.exp(-h_x)) + (1-y_train) * np.log(1+np.exp(h_x)))+ self.regularization(self.w[i - num1])
                    loss = y_train * np.log(1 + np.exp(-h_x)) + (1 - y_train) * np.log(
                        1 + np.exp(h_x)) + self.regularization(w_temp)
                    class_err = 1 - int(np.around(y_pred) == y_train)

                    if (inner_cnt <= tau - mu):
                        w_grad = x_train.T * (y_pred - y_train) + self.regularization.grad(w_temp)
                        self.w = self.w - learning_rate * w_grad  # 更新权值w

                    self.training_errors.append(loss)
                    self.class_errors.append(class_err)

                if ((self.temp_of_data + 1) % tau == 0 and self.is_private):
                    noise = np.random.laplace(loc=0.0, scale=sigma * learning_rate, size=(n_features + 1, 1))
                    # print("noise",noise)
                    self.w = self.w + noise

                inner_cnt += 1

                if (self.temp_of_data + 1) % tau == 0:
                    # print("communicating...")
                    # print("time:", self.temp_of_data)
                    # for k in range(len(self.w)):
                    #     if self.w[k, 0] > 1:
                    #         self.w[k, 0] = 1
                    #     if self.w[k, 0] < -1:
                    #         self.w[k, 0] = -1
                    self.w = projection(self.w, self.radius, is_l2_norm=True)
                    w_temp = self.w
                    batch_cnt += 1
                    inner_cnt = 1
                tt = tt + 1
            print("loss_mean:", sum(self.training_errors) / len(self.training_errors))
            # print("train_err_shape:",len(self.training_errors))
            print("class_err:", sum(self.class_errors) / len(self.class_errors))
            # print("class_err_shape:",len(self.class_errors))

    def predict(self, X):
        pass

    def calculate_weight(self):
        pass
