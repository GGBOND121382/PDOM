import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def comm_comp(stepsize, num_client, mu, d, num_of_neighbors, dst_name):
    eps_list = [0.1, 1.0, 10.0]
    # stepsize = 4000
    num_dataofclient_list = [i * stepsize for i in range(1, 6)]
    # num_client = 8
    # mu = 8
    # num_of_neighbors = 2
    # d = 117 + 1
    # dst_name = 'Comm.pdf'

    PDMA_sav = []
    DMA_sav = []
    PDMA_sav_strong = []

    for num_dataofclient in num_dataofclient_list:
        recursion = num_dataofclient * num_client

        tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        # print("tau_DMA:", tau_DMA)

        CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

        CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

        DMA_sav.append(CC_PDOCO / CC_DMA)

    DMA_sav = np.reshape(DMA_sav, (1, len(num_dataofclient_list)))

    for eps in eps_list:
        for num_dataofclient in num_dataofclient_list:
            recursion = num_dataofclient * num_client

            # tau_PDMA = 2 * mu + int((d * math.log(1 / delta) * num_dataofclient) ** (1. / 3) / (
            #         eps ** (2. / 3) * num_client ** (1. / 3)))
            tau_PDMA = 2 * mu + int(d ** (2. / 3) * num_dataofclient ** (1. / 3) / (
                    (num_client * eps) ** (2. / 3)))
            # print("tau_PDMA (epsilon):", tau_PDMA, eps)
            # tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

            # tau_PDMA_strong = mu + int(math.sqrt(d * math.log(1 / delta)) / (math.sqrt(num_client) * eps))
            tau_PDMA_strong = mu + int(d / (num_client * eps))

            CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

            CC_PDMA = num_dataofclient / tau_PDMA * 2 * (num_client - 1)

            # CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

            CC_PDMA_strong = num_dataofclient / tau_PDMA_strong * 2 * (num_client - 1)

            PDMA_sav.append(CC_PDOCO / CC_PDMA)
            # DMA_sav.append(CC_PDOCO / CC_DMA)
            PDMA_sav_strong.append(CC_PDOCO / CC_PDMA_strong)

    PDMA_sav = np.reshape(PDMA_sav, (3, len(num_dataofclient_list)))
    # DMA_sav = np.reshape(DMA_sav, (3, len(num_dataofclient_list)))
    PDMA_sav_strong = np.reshape(PDMA_sav_strong, (3, len(num_dataofclient_list)))

    DMA_sav = 1 / DMA_sav
    PDMA_sav = 1 / PDMA_sav
    PDMA_sav_strong = 1 / PDMA_sav_strong

    # labels = ["non-pri", "PDOM ($\\alpha = 0$)", "PDOM ($\\alpha =.01$)"]
    labels = ["non-pri", "PDOM ($\\alpha > 0$)", "PDOM ($\\alpha = 0$)"]
    # color = ["#4286F3", '#f4433c', '#ffbc32']
    # color = ["#4286F3", '#ffbc32', '#f4433c']
    # color = ["#4285F4", '#fbbc05', '#eb4334']
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    if (dst_name == 'POCO-diabetes-stoc-8.pdf'):
        plt.figure(figsize=(6.5, 3.7))
    else:
        plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    plt.yscale("log")

    num_dataofclient_list = np.array(num_dataofclient_list) / 10 ** 3

    if ('kddcup99' in dst_name):
        num_dataofclient_list = num_dataofclient_list.astype(dtype=int)
        num_dataofclient_list = num_dataofclient_list.astype(dtype=str)

    plt.plot(num_dataofclient_list, DMA_sav[0, :], linewidth=param_linewidth, marker="s",
             markersize=param_markersize_, c=color[0])

    plt.plot(num_dataofclient_list, PDMA_sav[0, :], '--', linewidth=param_linewidth, marker="o",
             markersize=param_markersize_, c=color[2])
    plt.plot(num_dataofclient_list, PDMA_sav[1, :], '-.', linewidth=param_linewidth, marker="v",
             markersize=param_markersize_, c=color[2])
    plt.plot(num_dataofclient_list, PDMA_sav[2, :], ':', linewidth=param_linewidth, marker="*",
             markersize=param_markersize_, c=color[2])
    # plt.plot(num_dataofclient_list, (POCO_lossmean[3, :] - Optimal_lossmean), marker="o", label="$\epsilon = 10$")

    #######################################################################################################

    # plt.plot(num_dataofclient_list, (POCOPL_nopri__lossmean - Optimal_lossmean), marker="o", markersize=param_markersize_, label="$non-pri$")
    plt.plot(num_dataofclient_list, PDMA_sav_strong[0, :], '--', linewidth=param_linewidth,
             marker="o", markersize=param_markersize_, c=color[1])
    plt.plot(num_dataofclient_list, PDMA_sav_strong[1, :], '-.', linewidth=param_linewidth,
             marker="v", markersize=param_markersize_, c=color[1])
    plt.plot(num_dataofclient_list, PDMA_sav_strong[2, :], ':', linewidth=param_linewidth,
             marker="*", markersize=param_markersize_, c=color[1])

    # plt.ylim(-0.025, 1.04)

    # 设置图例

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    legend1 = plt.legend(handles=patches, bbox_to_anchor=(1.03, 1.18), ncol=3, frameon=False, fontsize=13.5)
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
    plt.ylabel("Comm.-ratio", fontsize=15)
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format="pdf", bbox_inches="tight")
    plt.show()
    # print(DMA_sav)
    # print(PDMA_sav)
    # print(PDMA_sav_strong)


def comm_comp_topology(stepsize, num_client, mu_cyc, mu_cube, d, num_of_neighbors_cyc, num_of_neighbors_cube, dst_name):
    eps = 1.0
    # stepsize = 4000
    num_dataofclient_list = [i * stepsize for i in range(1, 6)]
    # num_client = 8
    # mu = 8
    # num_of_neighbors = 2
    # d = 117 + 1
    # dst_name = 'Comm.pdf'

    PDMA_sav = []
    DMA_sav = []
    PDMA_sav_strong = []

    mu = mu_cyc
    num_of_neighbors = num_of_neighbors_cyc

    print("cycle network")

    for num_dataofclient in num_dataofclient_list:
        recursion = num_dataofclient * num_client

        tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        print("tau_DMA:", tau_DMA)

        CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

        CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

        DMA_sav.append(CC_PDOCO / CC_DMA)

    # DMA_sav = np.reshape(DMA_sav, (1, len(num_dataofclient_list)))

    print()

    # for eps in eps_list:
    for num_dataofclient in num_dataofclient_list:
        recursion = num_dataofclient * num_client
        '''
        tau_PDMA = 2 * mu + int((d * math.log(1 / delta) * num_dataofclient) ** (1. / 3) / (
                eps ** (2. / 3) * num_client ** (1. / 3)))

        print("tau_PDMA:", tau_PDMA)

        # tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        tau_PDMA_strong = mu + int(math.sqrt(d * math.log(1 / delta)) / (math.sqrt(num_client) * eps))
        '''

        # tau_PDMA = 2 * mu + int((d * math.log(1 / delta) * num_dataofclient) ** (1. / 3) / (
        #         eps ** (2. / 3) * num_client ** (1. / 3)))
        tau_PDMA = 2 * mu + int(d ** (2. / 3) * num_dataofclient ** (1. / 3) / (
                (num_client * eps) ** (2. / 3)))
        # print("tau_PDMA (epsilon):", tau_PDMA, eps)
        # tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        # tau_PDMA_strong = mu + int(math.sqrt(d * math.log(1 / delta)) / (math.sqrt(num_client) * eps))
        tau_PDMA_strong = mu + int(d / (num_client * eps))

        CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

        CC_PDMA = num_dataofclient / tau_PDMA * 2 * (num_client - 1)

        # CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

        CC_PDMA_strong = num_dataofclient / tau_PDMA_strong * 2 * (num_client - 1)

        PDMA_sav.append(CC_PDOCO / CC_PDMA)
        # DMA_sav.append(CC_PDOCO / CC_DMA)
        PDMA_sav_strong.append(CC_PDOCO / CC_PDMA_strong)

    # PDMA_sav = np.reshape(PDMA_sav, (3, len(num_dataofclient_list)))
    # PDMA_sav_strong = np.reshape(PDMA_sav_strong, (3, len(num_dataofclient_list)))

    print()
    print("cube network")

    mu = mu_cube
    num_of_neighbors = num_of_neighbors_cube

    for num_dataofclient in num_dataofclient_list:
        recursion = num_dataofclient * num_client

        tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        print("tau_DMA:", tau_DMA)

        CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

        CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

        DMA_sav.append(CC_PDOCO / CC_DMA)

    # for eps in eps_list:
    for num_dataofclient in num_dataofclient_list:
        recursion = num_dataofclient * num_client

        '''
        tau_PDMA = 2 * mu + int((d * math.log(1 / delta) * num_dataofclient) ** (1. / 3) / (
                eps ** (2. / 3) * num_client ** (1. / 3)))

        print("tau_PDMA:", tau_PDMA)

        # tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        tau_PDMA_strong = mu + int(math.sqrt(d * math.log(1 / delta)) / (math.sqrt(num_client) * eps))
        '''

        # tau_PDMA = 2 * mu + int((d * math.log(1 / delta) * num_dataofclient) ** (1. / 3) / (
        #         eps ** (2. / 3) * num_client ** (1. / 3)))
        tau_PDMA = 2 * mu + int(d ** (2. / 3) * num_dataofclient ** (1. / 3) / (
                (num_client * eps) ** (2. / 3)))
        # print("tau_PDMA (epsilon):", tau_PDMA, eps)
        # tau_DMA = mu + math.ceil((mu * num_dataofclient / num_client) ** (1. / 3))

        # tau_PDMA_strong = mu + int(math.sqrt(d * math.log(1 / delta)) / (math.sqrt(num_client) * eps))
        tau_PDMA_strong = mu + int(d / (num_client * eps))

        CC_PDOCO = num_dataofclient * num_of_neighbors * num_client

        CC_PDMA = num_dataofclient / tau_PDMA * 2 * (num_client - 1)

        # CC_DMA = num_dataofclient / tau_DMA * 2 * (num_client - 1)

        CC_PDMA_strong = num_dataofclient / tau_PDMA_strong * 2 * (num_client - 1)

        PDMA_sav.append(CC_PDOCO / CC_PDMA)
        # DMA_sav.append(CC_PDOCO / CC_DMA)
        PDMA_sav_strong.append(CC_PDOCO / CC_PDMA_strong)

    DMA_sav = np.reshape(DMA_sav, (2, len(num_dataofclient_list)))
    PDMA_sav = np.reshape(PDMA_sav, (2, len(num_dataofclient_list)))
    PDMA_sav_strong = np.reshape(PDMA_sav_strong, (2, len(num_dataofclient_list)))

    DMA_sav = 1 / DMA_sav
    PDMA_sav = 1 / PDMA_sav
    PDMA_sav_strong = 1 / PDMA_sav_strong

    # print("PDMA_sav_strong:", PDMA_sav_strong)
    # labels = ["non-pri", "PDOM ($\\alpha = 0$)", "PDOM ($\\alpha =.01$)"]
    labels = ["non-pri", "PDOM ($\\alpha > 0$)", "PDOM ($\\alpha = 0$)"]
    # color = ["#4286F3", '#f4433c', '#ffbc32']
    # color = ["#4286F3", '#ffbc32', '#f4433c']
    # color = ["#4285F4", '#fbbc05', '#eb4334']
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    if (dst_name == 'POCO-diabetes-stoc-8.pdf'):
        plt.figure(figsize=(6.5, 3.7))
    else:
        plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    plt.yscale("log")

    num_dataofclient_list = np.array(num_dataofclient_list) / 10 ** 3

    if ('kddcup99' in dst_name):
        num_dataofclient_list = num_dataofclient_list.astype(dtype=int)
        num_dataofclient_list = num_dataofclient_list.astype(dtype=str)

    plt.plot(num_dataofclient_list, DMA_sav[0, :], '--', linewidth=param_linewidth, marker="o",
             markersize=param_markersize_, c=color[0])
    plt.plot(num_dataofclient_list, DMA_sav[1, :], '-.', linewidth=param_linewidth, marker="v",
             markersize=param_markersize_, c=color[0])

    plt.plot(num_dataofclient_list, PDMA_sav[0, :], '--', linewidth=param_linewidth, marker="o",
             markersize=param_markersize_, c=color[2])
    plt.plot(num_dataofclient_list, PDMA_sav[1, :], '-.', linewidth=param_linewidth, marker="v",
             markersize=param_markersize_, c=color[2])
    # plt.plot(num_dataofclient_list, (POCO_lossmean[3, :] - Optimal_lossmean), marker="o", label="$\epsilon = 10$")

    #######################################################################################################

    # plt.plot(num_dataofclient_list, (POCOPL_nopri__lossmean - Optimal_lossmean), marker="o", markersize=param_markersize_, label="$non-pri$")
    plt.plot(num_dataofclient_list, PDMA_sav_strong[0, :], '--', linewidth=param_linewidth,
             marker="o", markersize=param_markersize_, c=color[1])
    plt.plot(num_dataofclient_list, PDMA_sav_strong[1, :], '-.', linewidth=param_linewidth,
             marker="v", markersize=param_markersize_, c=color[1])

    # plt.ylim(-0.025, 1.04)

    # 设置图例

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    legend1 = plt.legend(handles=patches, bbox_to_anchor=(1.03, 1.18), ncol=3, frameon=False, fontsize=13.5)
    e1 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
                       label="cycle", linestyle="--")
    e2 = mlines.Line2D([], [], color='black', marker='v', markersize=9,
                       label="hypercube", linestyle="-.")
    plt.legend(handles=[e1, e2], bbox_to_anchor=(0.82, 1.12), ncol=2, frameon=False, fontsize=14)
    plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=15)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=15)
    plt.ylabel("Comm.-ratio", fontsize=15)
    plt.yticks([6 * 10 ** (-3), 10**(-2), 2 * 10 ** (-2)])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format="pdf", bbox_inches="tight")
    plt.show()
    # print(DMA_sav)
    # print(PDMA_sav)
    # print(PDMA_sav_strong)


if __name__ == '__main__':
    stepsize = 4000
    num_client = 8
    mu = 8
    num_of_neighbors = 2
    d = 117 + 1
    dst_name = 'Comm_kddcup99_cyc_8.pdf'
    comm_comp(stepsize, num_client, mu, d, num_of_neighbors, dst_name)

    stepsize = 2500
    num_client = 8
    mu = 8
    num_of_neighbors = 2
    d = 841 + 1
    dst_name = 'Comm_diabetes_cyc_8.pdf'
    comm_comp(stepsize, num_client, mu, d, num_of_neighbors, dst_name)

    stepsize = 4000
    num_client = 64
    # mu = 8
    # num_of_neighbors = 2
    mu_cyc = 64
    num_of_neighbors_cyc = 2
    mu_cube = 12
    num_of_neighbors_cube = 6
    d = 117 + 1
    dst_name = 'Comm_kddcup99_cyc_cube_64.pdf'
    comm_comp_topology(stepsize, num_client, mu_cyc, mu_cube, d, num_of_neighbors_cyc, num_of_neighbors_cube, dst_name)
