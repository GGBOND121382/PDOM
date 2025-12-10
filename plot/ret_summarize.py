import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import json
import sys

sys.path.append('../')

from optimization_utils.config_save_load import conf_load

# conf_dict = conf_load('../conf.ini')
# data = conf_dict['data']
# target = conf_dict['target']
# epsilon_list = conf_dict['eps_list']
# recursion_list = conf_dict['recur_list']
# rpt_times = conf_dict['rpt_times']
# number_of_clients = conf_dict['number_of_clients']

source, recursion_list = 'diabetes', [100000, ]
# source, recursion_list = 'kddcup99', [160000, ]
epsilon_list = [0.1, 1.0, 10.0]
# alpha_list = [0.0, 0.1, 0.01, 0.001, 0.0001]
alpha_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
topology = 'cycle_8'

fd_ret = open('class_err_ret', 'w')
avg_class_err_dict = {}

'''
############################################################
dirname = "../adv_setting/plot_data/"
#########################################
for recursion in recursion_list:
    for alpha in alpha_list:
        filename = dirname + 'DOCO_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + topology
        classerr_list = []
        with open(filename) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                classerr_list.append(float(tmp[1][:-1]))
        avg_class_err_dict['DOCO_' + repr(recursion) + '_' + repr(alpha)] = np.mean(classerr_list)

#########################################
for epsilon in epsilon_list:
    for recursion in recursion_list:
        for alpha in alpha_list:
            filename = dirname + 'PDOCO_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            classerr_list = []
            with open(filename) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    classerr_list.append(float(tmp[1][:-1]))
            avg_class_err_dict['PDOCO_' + repr(recursion) + '_' + repr(alpha) + '_' + repr(epsilon)] = np.mean(classerr_list)

#########################################
for epsilon in epsilon_list:
    for recursion in recursion_list:
        for alpha in alpha_list:
            filename = dirname + 'PDOMO_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            classerr_list = []
            with open(filename) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    classerr_list.append(float(tmp[1][:-1]))
            avg_class_err_dict['PDOMO_' + repr(recursion) + '_' + repr(alpha) + '_' + repr(epsilon)] = np.mean(classerr_list)
'''

############################################################
dirname = "../iid_setting/plot_data/"
#########################################
for recursion in recursion_list:
    for alpha in alpha_list:
        filename = dirname + 'non_pri_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + topology
        classerr_list = []
        with open(filename) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                classerr_list.append(float(tmp[1][:-1]))
        avg_class_err_dict['non_pri_' + repr(recursion) + '_' + repr(alpha)] = np.mean(classerr_list)
#########################################
for epsilon in epsilon_list:
    for recursion in recursion_list:
        for alpha in alpha_list:
            filename = dirname + 'PDOMS_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            classerr_list = []
            with open(filename) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    classerr_list.append(float(tmp[1][:-1]))
            avg_class_err_dict['PDOMS_' + repr(recursion) + '_' + repr(alpha) + '_' + repr(epsilon)] = np.mean(classerr_list)
#########################################
for epsilon in epsilon_list:
    for recursion in recursion_list:
        for alpha in [0.0, ]:
            filename = dirname + 'non_comm_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            classerr_list = []
            with open(filename) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    classerr_list.append(float(tmp[1][:-1]))
            avg_class_err_dict['PFTAL_' + repr(recursion) + '_' + repr(alpha) + '_' + repr(epsilon)] = np.mean(classerr_list)
        for alpha in alpha_list[1:]:
            filename = dirname + 'PFTAL_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            classerr_list = []
            with open(filename) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    classerr_list.append(float(tmp[1][:-1]))
            avg_class_err_dict['PFTAL_' + repr(recursion) + '_' + repr(alpha) + '_' + repr(epsilon)] = np.mean(classerr_list)

json.dump(avg_class_err_dict, fd_ret)
