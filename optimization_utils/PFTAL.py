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


class NoiseTree:
    def __init__(self, std, dimen):
        self.std = std
        self.dimen = dimen
        self.step = 0
        self.binary = [0]
        self.noise_sum = np.zeros(dimen)
        self.recorded = [np.zeros(dimen), ]

    def __call__(self):
        if self.std == 0:
            return self.noise_sum

        self.step += 1
        idx = 0

        '''
        if (math.log(self.step, 2) in range(10)):
            print("before before...")
            #print(self.binary)
            print(self.noise_sum)
            #print(self.recorded)
        '''
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            # for ns, re in zip(self.noise_sum, self.recorded[idx]):
            #    ns -= re
            '''
            for i in range(len(self.noise_sum)):
                self.noise_sum[i] -= self.recorded[idx][i]
                # print("i:", self.recorded[idx][i])
            '''
            self.noise_sum -= self.recorded[idx]            # equivalent to the former commented codes
            # print("noise shape: ", self.noise_sum.shape, self.recorded[idx].shape)
            idx += 1
        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append(np.zeros(self.dimen))

        '''
        if(math.log(self.step, 2) in range(10)):
            print("before...")
            #print(self.binary)
            print(self.noise_sum)
        '''

        '''
        for i in range(len(self.noise_sum)):
            n = np.random.laplace(0, self.std, 1)
            # n = i + 1
            self.noise_sum[i] += n
            self.recorded[idx][i] = n
            # re = n
        '''

        ######################################### equivalent to the former codes
        n = np.random.laplace(0, self.std, self.noise_sum.shape)
        # print("noise shape: ", n.shape, self.noise_sum.shape, self.recorded[idx].shape)
        # print()
        self.noise_sum += n
        self.recorded[idx] = n
        #########################################

        self.binary[idx] = 1
        '''
        if (math.log(self.step, 2) in range(10)):
            print("after...")
            #print(self.binary)
            print(self.noise_sum)
            print()
            print()
        '''
        # print self.binary
        # print
        # print(self.noise_sum)
        return self.noise_sum


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


class PFTAL():
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
                 num_client=8, alpha=0.0, is_private=True, is_lagged=True, radius=10.):
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
        self.w = []
        self.v = []
        self.w_all_sum = []
        for i in range(self.num_client):
            # w = np.random.uniform(0, 0, (n_features, 1))
            # w = np.random.uniform(0, 0, (n_features, 1))
            # b = 0
            # w = np.random.uniform(0, 0, (n_features + 1, 1))
            w = np.random.uniform(-self.radius, self.radius, (n_features + 1, 1))
            w = projection(w, radius=self.radius, is_l2_norm=True)
            self.w.append(w)
            self.v.append(np.zeros((n_features + 1, 1)))
            self.w_all_sum.append(np.zeros((n_features + 1, 1)))
        # print("Intial...")
        # print(self.w[0])
        # self.w = w

    def fit(self, X, y):
        m_samples, n_features = X.shape
        num_dataofclient = int(m_samples / self.num_client)
        self.initialize_weights(n_features)
        # print(X.shape)
        X = np.insert(X, 0, 1, axis=1)
        # print(X.shape)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = np.zeros(self.num_client)
        self.class_errors = np.zeros(self.num_client)
        # w_temp = self.w[0]
        w_temp = copy.deepcopy(self.w)
        # print("w_temp:", w_temp.size)
        tt = 0
        # L_G = 1. / 2 + self.alpha  ## L_G equals to the largest singlular value of Hessian of regularized logistic loss, which equals to 1/2 + alpha
        # q = 2 * L_G ** 2 / self.alpha ** 2 - 1
        # L_cons = math.sqrt(2) + self.alpha * math.sqrt(n_features + 1)
        L_cons = math.sqrt(2) + self.alpha * self.radius
        d = n_features + 1
        # C_norm = math.sqrt(d)
        C_norm = self.radius
        sigma = 2 * L_cons * math.sqrt(d) * math.ceil(math.log(num_dataofclient, 2)) / self.eps  # in PFTAL, sigma depends on log(T, 2), the sensitivity for the sum of gradients is 2L
        print("noise sigma:", sigma)
        noise_tree = []
        for i in range(self.num_client):
            noise_tree.append(NoiseTree(sigma, w_temp[0].shape))

        if(self.alpha > 0):
            if self.gradient == True:
                # 梯度下降
                while True:
                    self.temp_of_data = tt
                    if self.temp_of_data == num_dataofclient:
                        break
                    num1 = self.num_client * (self.temp_of_data)
                    num2 = self.num_client * (self.temp_of_data + 1)
                    for i in range(num1, num2):
                        self.w_all_sum[i - num1] += self.w[i - num1]
                        x_train = np.reshape(X[i], (1, len(X[i])))
                        y_train = y[i]

                        # the update procedure
                        h_x = x_train.dot(self.w[i - num1])
                        h_x = np.array(h_x)
                        y_pred = sigmoid(h_x[0, 0])

                        w_grad = x_train.T * (y_pred - y_train) + self.regularization.grad(self.w[i - num1])
                        self.v[i - num1] += w_grad
                        random_noise = noise_tree[i - num1]()
                        noisy_v = self.v[i - num1] + random_noise
                        self.w[i - num1] = (self.alpha * self.w_all_sum[i - num1] - noisy_v) / (
                                self.alpha * (self.temp_of_data + 1))  ## solve the quadratic optimization problem
                        # for k in range(len(self.w[i - num1])):
                        #     if self.w[i - num1][k, 0] > 1:
                        #         self.w[i - num1][k, 0] = 1
                        #     if self.w[i - num1][k, 0] < -1:
                        #         self.w[i - num1][k, 0] = -1
                        self.w[i - num1] = projection(self.w[i - num1], self.radius, is_l2_norm=True)

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

                    w_temp = copy.deepcopy(self.w)  # we can modify "0" to some other index to control the reference learner
                    tt = tt + 1

                self.training_errors /= (self.num_client * num_dataofclient)
                self.class_errors /= (self.num_client * num_dataofclient)

                print("loss_mean:", self.training_errors)
                print("class_err:", self.class_errors)

        else:
            LHS = 2 * num_dataofclient * sigma * math.sqrt(2 * d * math.log(num_dataofclient, 2))
            RHS = C_norm ** 2
            param_lambda = math.sqrt(LHS / RHS)
            if self.gradient == True:
                # 梯度下降
                while True:
                    self.temp_of_data = tt
                    if self.temp_of_data == num_dataofclient:
                        break
                    num1 = self.num_client * (self.temp_of_data)
                    num2 = self.num_client * (self.temp_of_data + 1)
                    for i in range(num1, num2):
                        self.w_all_sum[i - num1] += self.w[i - num1]
                        x_train = np.reshape(X[i], (1, len(X[i])))
                        y_train = y[i]

                        # the update procedure
                        h_x = x_train.dot(self.w[i - num1])
                        h_x = np.array(h_x)
                        # print("h_x shape:", h_x.shape)
                        y_pred = sigmoid(h_x[0, 0])

                        w_grad = x_train.T * (y_pred - y_train)
                        self.v[i - num1] += w_grad
                        random_noise = noise_tree[i - num1]()
                        noisy_v = self.v[i - num1] + random_noise
                        self.w[i - num1] = - noisy_v / param_lambda  ## solve the quadratic optimization problem
                        # for k in range(len(self.w[i - num1])):
                        #     # print("type of w:",type(self.w))
                        #     # print(self.w[j][k, 0])
                        #     if self.w[i - num1][k, 0] > 1:
                        #         self.w[i - num1][k, 0] = 1
                        #     if self.w[i - num1][k, 0] < -1:
                        #         self.w[i - num1][k, 0] = -1
                        self.w[i - num1] = projection(self.w[i - num1], self.radius, is_l2_norm=True)

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

                    w_temp = copy.deepcopy(
                        self.w)  # we can modify "0" to some other index to control the reference learner
                    tt = tt + 1
                self.training_errors /= (self.num_client * num_dataofclient)
                self.class_errors /= (self.num_client * num_dataofclient)

                print("loss_mean:", self.training_errors)
                print("class_err:", self.class_errors)

    def predict(self, X):
        pass

    def calculate_weight(self):
        pass
