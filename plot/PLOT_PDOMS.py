import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import sys
sys.path.append('../')

from optimization_utils.config_save_load import conf_load


def regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology, alpha_dict=None):
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    if ('diabetes-8' in dst_name):
        plt.figure(figsize=(6.5, 3.7))
    else:
        plt.figure(figsize=(6.5, 3.85))
    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # PDOM
    PDOM_class_err_mean = []
    if(alpha_dict==None):
        alpha = 0.0
    else:
        alpha = alpha_dict['PDOMS']
    for epsilon in eps_list:
        for recursion in recur_list:
            file_name = src_dir_name + '/PDOMS_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    class_err_list.append(float(tmp[1][:-1]))
            PDOM_class_err_mean.append(np.mean(class_err_list))
    PDOM_class_err_mean = np.reshape(PDOM_class_err_mean, (len(eps_list), len(recur_list)))
    ############################################################
    # PFTAL
    PFTAL_class_err_mean = []
    if (alpha_dict == None):
        alpha = 0.0
        filename_pre = '/non_comm_'
    else:
        alpha = alpha_dict['non_comm']
        filename_pre = '/PFTAL_'
    for epsilon in eps_list:
        for recursion in recur_list:
            file_name = src_dir_name + filename_pre + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' +\
                            repr(epsilon) + '_' + topology
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    class_err_list.append(float(tmp[1][:-1]))
            PFTAL_class_err_mean.append(np.mean(class_err_list))
    PFTAL_class_err_mean = np.reshape(PFTAL_class_err_mean, (len(eps_list), len(recur_list)))
    ############################################################
    # non_pri
    non_pri_class_err_mean = []
    if (alpha_dict == None):
        alpha = 0.0
    else:
        alpha = alpha_dict['non_pri']
    for recursion in recur_list:
        file_name = src_dir_name + '/non_pri_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + topology
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                class_err_list.append(float(tmp[1][:-1]))
        non_pri_class_err_mean.append(np.mean(class_err_list))
    non_pri_class_err_mean = np.array(non_pri_class_err_mean)
    ############################################################

    recur_list = np.array(recur_list) / (number_of_clients * 10 ** 3)
    if ('kddcup99' in dst_name):
        recur_list = recur_list.astype(dtype=int)
        recur_list = recur_list.astype(dtype=str)

    # print(Optimal_lossmean)

    #######################################################################################################

    plt.plot(recur_list, non_pri_class_err_mean, linewidth=param_linewidth, marker="s",
             markersize=param_markersize_, c=color[0])

    plt.plot(recur_list, PFTAL_class_err_mean[0, :], '--', linewidth=param_linewidth,
             marker="o", markersize=param_markersize_, c=color[1])
    plt.plot(recur_list, PFTAL_class_err_mean[1, :], '-.', linewidth=param_linewidth,
             marker="v", markersize=param_markersize_, c=color[1])
    plt.plot(recur_list, PFTAL_class_err_mean[2, :], ':', linewidth=param_linewidth,
             marker="*", markersize=param_markersize_, c=color[1])
    # plt.plot(recur_list, (POCO_lossmean[3, :] - Optimal_lossmean), marker="o", label="$\epsilon = 10$")

    #######################################################################################################

    # plt.plot(recur_list, (POCOPL_nopri__lossmean - Optimal_lossmean), marker="o", markersize=param_markersize_, label="$non-pri$")
    plt.plot(recur_list, PDOM_class_err_mean[0, :], '--', linewidth=param_linewidth, marker="o",
             markersize=param_markersize_, c=color[2])
    plt.plot(recur_list, PDOM_class_err_mean[1, :], '-.', linewidth=param_linewidth, marker="v",
             markersize=param_markersize_, c=color[2])
    plt.plot(recur_list, PDOM_class_err_mean[2, :], ':', linewidth=param_linewidth, marker="*",
             markersize=param_markersize_, c=color[2])

    # plt.ylim(-0.025, 1.04)

    # 设置图例
    labels = ["non-pri", "non-comm", "PDOM"]

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.97, 1.18), ncol=3, frameon=False, fontsize=14)
    e1 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
                       label="$\epsilon = 0.1$", linestyle="--")
    e2 = mlines.Line2D([], [], color='black', marker='v', markersize=9,
                       label="$\epsilon = 1$", linestyle="-.")
    e3 = mlines.Line2D([], [], color='black', marker='*', markersize=9,
                       label="$\epsilon = 10$", linestyle=":")
    plt.legend(handles=[e1, e2, e3], bbox_to_anchor=(0.95, 1.12), ncol=3, frameon=False, fontsize=14)
    plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=15)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=15)
    plt.ylabel("Error rates", fontsize=15)
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format="pdf", bbox_inches="tight")
    plt.show()


def regret_plot_64(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology_list, alpha_dict=None):
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))
    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # PDOM
    PDOM_class_err_mean = []
    if (alpha_dict == None):
        alpha = 0.0
    else:
        alpha = alpha_dict['PDOMS']
    for topology in topology_list:
        for epsilon in eps_list:
            for recursion in recur_list:
                file_name = src_dir_name + '/PDOMS_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + \
                            repr(epsilon) + '_' + topology
                class_err_list = []
                with open(file_name) as fd:
                    for line in fd.readlines():
                        tmp = line.split(" ")
                        class_err_list.append(float(tmp[1][:-1]))
                PDOM_class_err_mean.append(np.mean(class_err_list))
    PDOM_class_err_mean = np.reshape(PDOM_class_err_mean, (len(topology_list), len(recur_list)))
    ############################################################
    # PFTAL
    PFTAL_class_err_mean = []
    if (alpha_dict == None):
        alpha = 0.0
        filename_pre = '/non_comm_'
    else:
        alpha = alpha_dict['non_comm']
        filename_pre = '/PFTAL_'
    for topology in ['cycle_64']:
        for epsilon in eps_list:
            for recursion in recur_list:
                file_name = src_dir_name + filename_pre + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + \
                            repr(epsilon) + '_' + topology
                class_err_list = []
                with open(file_name) as fd:
                    for line in fd.readlines():
                        tmp = line.split(" ")
                        class_err_list.append(float(tmp[1][:-1]))
                PFTAL_class_err_mean.append(np.mean(class_err_list))
    PFTAL_class_err_mean = np.reshape(PFTAL_class_err_mean, (1, len(recur_list)))
    ############################################################
    # non_pri
    non_pri_class_err_mean = []
    if (alpha_dict == None):
        alpha = 0.0
        topology_list_tmp = topology_list
    else:
        alpha = alpha_dict['non_pri']
        topology_list_tmp = ['cycle_64']
    for topology in topology_list_tmp:
        for recursion in recur_list:
            file_name = src_dir_name + '/non_pri_' + source + '_' + repr(alpha) + '_' + repr(recursion) + '_' + topology
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    class_err_list.append(float(tmp[1][:-1]))
            non_pri_class_err_mean.append(np.mean(class_err_list))
    non_pri_class_err_mean = np.reshape(non_pri_class_err_mean, (len(topology_list_tmp), len(recur_list)))

    recur_list = np.array(recur_list) / (number_of_clients * 10 ** 3)
    if ('kddcup99' in dst_name):
        recur_list = recur_list.astype(dtype=int)
        recur_list = recur_list.astype(dtype=str)

    # print(Optimal_lossmean)

    #######################################################################################################

    plt.plot(recur_list, non_pri_class_err_mean[0, :], '--', linewidth=param_linewidth,
             marker="o", markersize=param_markersize_, c=color[0])

    if(alpha == 0):
        plt.plot(recur_list, non_pri_class_err_mean[1, :], '-.', linewidth=param_linewidth,
                 marker="v", markersize=param_markersize_, c=color[0])

    plt.plot(recur_list, PFTAL_class_err_mean[0, :], '--', linewidth=param_linewidth,
             marker="o", markersize=param_markersize_, c=color[1])
    # plt.plot(recur_list, PFTAL_class_err_mean[1, :], '-.', linewidth=param_linewidth,
    #          marker="v", markersize=param_markersize_, c=color[1])
    # plt.plot(recur_list, (POCOPL_pal_tau1_lossmean[2, :] - Optimal_lossmean), ':', linewidth = param_linewidth, marker="*", markersize=param_markersize_, c=color[1])
    # plt.plot(recur_list, (POCO_lossmean[3, :] - Optimal_lossmean), marker="o", label="$\epsilon = 10$")

    #######################################################################################################

    # plt.plot(recur_list, (POCOPL_nopri__lossmean - Optimal_lossmean), marker="o", markersize=param_markersize_, label="$non-pri$")
    plt.plot(recur_list, PDOM_class_err_mean[0, :], '--', linewidth=param_linewidth, marker="o",
             markersize=param_markersize_, c=color[2])
    plt.plot(recur_list, PDOM_class_err_mean[1, :], '-.', linewidth=param_linewidth, marker="v",
             markersize=param_markersize_, c=color[2])
    # plt.plot(recur_list, (POCOPL_pal_lossmean[2, :] - Optimal_lossmean), ':', linewidth = param_linewidth, marker="*", markersize=param_markersize_, c=color[2])

    # plt.ylim(-0.025, 1.04)

    # 设置图例
    labels = ["non-pri", "non-comm", "PDOM"]

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.97, 1.18), ncol=3, frameon=False, fontsize=14)
    e1 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
                       label="cycle", linestyle="--")
    e2 = mlines.Line2D([], [], color='black', marker='v', markersize=9,
                       label="hypercube", linestyle="-.")
    # e3 = mlines.Line2D([], [], color='black', marker='*',markersize=9,
    #                       label="$\epsilon = 1$",linestyle=":")
    plt.legend(handles=[e1, e2], bbox_to_anchor=(0.82, 1.12), ncol=2, frameon=False, fontsize=14)
    plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=15)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=15)
    plt.ylabel("Error rates", fontsize=15)

    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format="pdf", bbox_inches="tight")
    plt.show()


if (__name__ == '__main__'):
    data_source_list = ['diabetes', 'kddcup99']
    for data_source in data_source_list:
        conf_dict = conf_load('conf-dir/conf-stoc-' + data_source + '-8.ini')
        source = conf_dict['source']
        data = conf_dict['data']
        target = conf_dict['target']
        # eps_list = conf_dict['eps_list']
        eps_list = [0.1, 1.0, 10.0]
        recur_list = conf_dict['recur_list']
        rpt_times = conf_dict['rpt_times']
        topology = conf_dict['topology']
        is_minus_one = conf_dict['is_minus_one']
        # alpha_list = conf_dict['alpha_list']
        alpha_dict = conf_dict['alpha_dict']
        number_of_clients = conf_dict['number_of_clients']

        src_dir_name = '../iid_setting/plot_data'
        dst_name = 'PDOMS-' + data_source + '-8-cvx.pdf'
        regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology)

        src_dir_name = '../iid_setting/plot_data'
        dst_name = 'PDOMS-' + data_source + '-8-strong.pdf'
        regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology, alpha_dict)

    data_source = 'kddcup99'
    conf_dict = conf_load('conf-dir/conf-stoc-' + data_source + '-64.ini')
    source = conf_dict['source']
    data = conf_dict['data']
    target = conf_dict['target']
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    rpt_times = conf_dict['rpt_times']
    topology = conf_dict['topology']
    is_minus_one = conf_dict['is_minus_one']
    # alpha_list = conf_dict['alpha_list']
    alpha_dict = conf_dict['alpha_dict']
    number_of_clients = conf_dict['number_of_clients']
    topology_list = ['cycle_64', 'cube_64']

    src_dir_name = '../iid_setting/plot_data'
    dst_name = 'PDOMS-' + data_source + '-64-cvx.pdf'
    regret_plot_64(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology_list)

    src_dir_name = '../iid_setting/plot_data'
    dst_name = 'PDOMS-' + data_source + '-64-strong.pdf'
    regret_plot_64(eps_list, recur_list, number_of_clients, src_dir_name, dst_name, topology_list, alpha_dict)



    '''
    ############################################################################
    conf_dict = conf_load('../config/conf_kddcup99_64.ini')
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    number_of_clients = conf_dict['number_of_clients']
    ############################################################################
    src_dir_name_cyc = '../POCO-kddcup99-64/data-credit-logR'
    src_dir_name_cube = '../POCO-kddcup99-64-cube/data-credit-logR'
    dst_name = 'POCO-kddcup99-64-cyc-cube.pdf'
    regret_plot_64(eps_list, recur_list, number_of_clients, src_dir_name_cyc, src_dir_name_cube, dst_name)
    ############################################################################
    src_dir_name_cyc = '../POCO-kddcup99-64-conv/data-credit-logR'
    src_dir_name_cube = '../POCO-kddcup99-64-cube-conv/data-credit-logR'
    dst_name = 'POCO-kddcup99-64-conv-cyc-cube.pdf'
    regret_plot_64(eps_list, recur_list, number_of_clients, src_dir_name_cyc, src_dir_name_cube, dst_name)
    ############################################################################

    ############################################################################
    conf_dict = conf_load('../config/conf_kddcup99_8.ini')
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    number_of_clients = conf_dict['number_of_clients']
    ############################################################################
    src_dir_name = '../POCO-kddcup99-8/data-credit-logR'
    dst_name = 'POCO-kddcup99-8.pdf'
    regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name)
    ############################################################################
    src_dir_name = '../POCO-kddcup99-8-conv/data-credit-logR'
    dst_name = 'POCO-kddcup99-8-conv.pdf'
    regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name)
    ############################################################################

    ############################################################################
    conf_dict = conf_load('../config/conf_diabetes_8.ini')
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    number_of_clients = conf_dict['number_of_clients']
    ############################################################################
    src_dir_name = '../POCO-diabetes-8/data-credit-logR'
    dst_name = 'POCO-diabetes-8.pdf'
    regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name)
    ############################################################################
    src_dir_name = '../POCO-diabetes-8-conv/data-credit-logR'
    dst_name = 'POCO-diabetes-8-conv.pdf'
    regret_plot(eps_list, recur_list, number_of_clients, src_dir_name, dst_name)
    ############################################################################
    '''