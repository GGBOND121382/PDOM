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


class PDOMO():

    def __init__(self, learning_rate=0.0001, regularization=l2_regularization, gradient=True, eps=None,
                 num_client=8, alpha=0., is_private=True, is_lagged=True, is_comm=True, radius=10.):
        # self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.eps = eps
        self.num_client = num_client
        self.alpha = alpha
        self.temp_of_data = 0
        self.is_private = is_private
        self.is_lagged = is_lagged
        self.is_comm = is_comm
        self.radius = radius
        self.w = []
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization(alpha=self.alpha)

    def initialize_weights(self, n_features):
        # 初始化参数
        self.w = []
        for i in range(self.num_client):
            w = np.random.uniform(-self.radius, self.radius, (n_features + 1, 1))
            w = projection(w, radius=self.radius, is_l2_norm=True)
            self.w.append(w)

    def fit(self, X, y, A, selected_learner=None):
        m_samples, n_features = X.shape
        num_dataofclient = int(m_samples / self.num_client)
        self.initialize_weights(n_features)

        positive_samples = np.sum(y)
        print("positive percentage:", positive_samples / m_samples)

        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = np.zeros(self.num_client)
        self.class_errors = np.zeros(self.num_client)
        w_temp = copy.deepcopy(self.w)
        w_store = copy.deepcopy(self.w)  # store the lagged model

        tt = 0
        L_G = 1. / 2 + self.alpha  ## L_G equals to the largest singlular value of Hessian of regularized logistic loss, which equals to 1/2 + alpha
        L_cons = math.sqrt(2) + self.alpha * self.radius
        d = n_features + 1
        if (self.is_private and self.is_lagged):
            tau = max(int(d / self.eps), 1)
        else:
            tau = 1
        print("tau:", tau)
        batch_cnt = 1
        interval_index = 0
        interval_length = int(num_dataofclient / 3)
        if self.gradient == True:
            # 梯度下降
            while True:
                self.temp_of_data = tt
                interval_index = int(tt / interval_length)
                if self.temp_of_data == num_dataofclient:
                    break
                num1 = self.num_client * (self.temp_of_data)
                num2 = self.num_client * (self.temp_of_data + 1)
                for i in range(num1, num2):
                    currentLearner = i - num1

                    if (self.is_private):
                        S = 2 * L_cons * math.sqrt(d)
                        sigma = S / self.eps
                    else:
                        sigma = 0
                    if (self.is_private and self.alpha > 0):
                        learning_rate = 1 / (self.alpha * batch_cnt * tau)
                    elif (self.is_private and self.alpha == 0 and self.is_lagged):
                        # learning_rate = math.sqrt(self.eps) / math.sqrt(d * num_dataofclient)
                        learning_rate = 1 / math.sqrt(num_dataofclient)
                    elif(self.is_private and self.alpha == 0):
                        learning_rate = 1 / math.sqrt(num_dataofclient)
                    elif(self.alpha == 0):
                        learning_rate = 1 / (math.sqrt(
                            num_dataofclient))  # specified in "Distributed Autonomous Online Learning Regrets and Intrinsic Privacy-Preserving Properties"
                    elif(self.alpha > 0):
                        learning_rate = 1 / (self.alpha * (
                                    1 + self.temp_of_data))  # specified in "Distributed Autonomous Online Learning Regrets and Intrinsic Privacy-Preserving Properties"
                    else:
                        assert False

                    sigma *= learning_rate

                    x_train = np.reshape(X[i], (1, len(X[i])))
                    y_train = y[i]

                    if (interval_index % 2 == 1):
                        y_train = 1 - y_train

                    # the update procedure
                    h_x = x_train.dot(w_store[currentLearner])
                    h_x = np.array(h_x)
                    # print("h_x shape:", h_x.shape)
                    y_pred = sigmoid(h_x[0, 0])
                    # loss = np.mean(y_train * np.log(1+ np.exp(-h_x)) + (1-y_train) * np.log(1+np.exp(h_x)))+ self.regularization(self.w[currentLearner])

                    w_grad = x_train.T * (y_pred - y_train) + self.regularization.grad(w_store[currentLearner])
                    self.w[currentLearner] = self.w[currentLearner] - learning_rate * w_grad
                    # print("weight:",self.w[0][0])
                    # add noise

                    # compute the loss corresponding to each model
                    for selected_learner in range(self.num_client):
                        h_x = x_train.dot(w_temp[selected_learner])
                        # print("h_x shape:", h_x.shape)
                        y_pred = sigmoid(h_x[0, 0])
                        loss = y_train * np.log(1 + np.exp(-h_x)) + (1 - y_train) * np.log(
                            1 + np.exp(h_x)) + self.regularization(w_temp[selected_learner])

                        class_err = 1 - int(np.around(y_pred) == y_train)
                        self.training_errors[selected_learner] += loss
                        self.class_errors[selected_learner] += class_err

                    if (self.temp_of_data + 1) % tau == 0:
                        noise = np.random.laplace(loc=0.0, scale=sigma, size=(d, 1))
                        # print("noise",noise)
                        self.w[currentLearner] = self.w[currentLearner] + noise
                        self.w[currentLearner] = projection(self.w[currentLearner], self.radius, is_l2_norm=True)

                if (self.temp_of_data + 1) % tau == 0:
                    # print("t:", self.temp_of_data + 1)
                    w_temp = copy.deepcopy(self.w)
                    # #average the perturbed model
                    w_store = copy.deepcopy(self.w)
                    if (self.is_comm):
                        self.w = []
                        for i in range(self.num_client):
                            temp = 0
                            for j in range(self.num_client):
                                temp += A[j, i] * w_store[j]
                            self.w.append(temp)
                    batch_cnt += 1
                tt = tt + 1

            self.training_errors /= (self.num_client * num_dataofclient)
            self.class_errors /= (self.num_client * num_dataofclient)

            print("loss_mean:", self.training_errors)
            print("class_err:", self.class_errors)
