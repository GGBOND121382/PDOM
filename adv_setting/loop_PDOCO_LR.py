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

from optimization_utils.PDOMO import PDOMO

from optimization_utils.config_save_load import conf_load

from optimization_utils.generate_gossip_matrix_A import two2three_cycle, two2six_cycle, hyper_cube


def run():
    conf_dict = conf_load()
    source = conf_dict['source']
    data = conf_dict['data']
    target = conf_dict['target']
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    rpt_times = conf_dict['rpt_times']
    topology = conf_dict['topology']
    A = conf_dict['A']
    A = np.array(A)
    is_minus_one = conf_dict['is_minus_one']
    number_of_clients = conf_dict['number_of_clients']
    alpha_dict = conf_dict['alpha_dict']

    X = torch.load(data, weights_only=False)
    # X = np.array(X)
    X = MaxAbsScaler().fit_transform(X)
    X = preprocessing.normalize(X, norm='l2')
    y = torch.load(target, weights_only=False)
    y = np.array(y, dtype=int)
    if is_minus_one:
        y = (y + 1) / 2  # in kddcup99, y\in \{-1, 1\}. We need to transform them into \{0, 1\}.
    y = np.array(y, dtype=int)
    y = np.reshape(y, (-1, 1))
    alpha_list = conf_dict['alpha_list']
    print(X.shape)
    print(y.shape)
    print(np.sum(y))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, seed=1)
    X_train = X
    y_train = y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001)
    
    # for epsilon in [0.01, 1]:
    #   for recursion in [6000, 12000, 18000, 24000, 29990]:

    alpha_g = alpha_dict['PDOCO']
    # for alpha in alpha_list:
    for alpha in [0.0, alpha_g]:
        for epsilon in eps_list:
            # for recursion in [10, 6000, 12000, 18000, 24000, 29990]:
            for recursion in recur_list:
                file_name = './plot_data/PDOCO_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
                fd = open(file_name, 'a')
                unit_Result = []

                for rpt in range(rpt_times):
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001)

                    print(str(epsilon) + ' ' + str(recursion) + ' ' + str(rpt))

                    # print("test:",sigmoid(np.array([[0.1]])))
                    delta = 1. / (recursion ** 2)
                    clf = PDOMO(eps=epsilon, is_lagged=False, num_client=number_of_clients, alpha=alpha)
                    # A = two2three_cycle()

                    clf.fit(X_train[:recursion, :], y_train[:recursion, :], A)
                    loss_mean_list = clf.training_errors.tolist()
                    class_err_list = clf.class_errors.tolist()

                    # print(clf.w)
                    for loss_mean, class_err in zip(loss_mean_list, class_err_list):
                        fd.write(repr(loss_mean) + ' ' + repr(class_err) + '\n')
                    # unit_Result.append([loss_mean, class_err])
                fd.close()


if __name__ == '__main__':
    run()
