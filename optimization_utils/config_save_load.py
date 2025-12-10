import json
import numpy

from optimization_utils.generate_gossip_matrix_A import two2three_cycle, hyper_cube, two2six_cycle


def conf_save(source, data, target, eps_list=[1., ], recur_list=[70000, ], rpt_times=100, number_of_clients=8,
              topology='cycle_8', A=two2three_cycle(), mu=8, alpha_list=[0.,], is_minus_one=False,
              alpha_dict={}, filename='conf.ini'):
    with open(filename, 'w') as fd:
        conf_dict = {'source': source, 'data': data, 'target': target, 'eps_list': eps_list, 'recur_list': recur_list,
                     'rpt_times': rpt_times, 'number_of_clients': number_of_clients, 'topology': topology,
                     'A': A.tolist(), 'mu': mu, 'alpha_list': alpha_list, 'is_minus_one': is_minus_one,
                     'alpha_dict': alpha_dict}
        json.dump(conf_dict, fd)


def conf_load(filename='conf.ini'):
    with open(filename) as fd:
        conf_dict = json.load(fd)
    return conf_dict


if __name__ == '__main__':
    data_source = 'diabetes'
    # data_source = 'kddcup99'

    # num_of_clients, topology, A, mu = 64, 'cycle_64', two2six_cycle(), 64
    # num_of_clients, topology, A, mu = 64, 'cube_64', hyper_cube(6), 12
    num_of_clients, topology, A, mu = 8, 'cycle_8', two2three_cycle(), 8
    setting_list = ['adv_setting', 'iid_setting']

    # if(data_source == 'diabetes'):
    #     alpha_dict = {'DOCO': 0.1, 'PDOCO': 0.1, 'PDOMO': 0.1, 'non_pri': 0.1, 'non_comm': 0.001, 'PDOMS': 0.1}
    # else:
    #     alpha_dict = {'DOCO': 0.001, 'PDOCO': 0.1, 'PDOMO': 0.1, 'non_pri': 0.001, 'non_comm': 0.0001, 'PDOMS': 0.01}

    if (data_source == 'diabetes'):
        alpha_dict = {'DOCO': 0.001, 'PDOCO': 0.1, 'PDOMO': 0.01, 'non_pri': 0.1, 'non_comm': 0.001, 'PDOMS': 0.1}
    else:
        alpha_dict = {'DOCO': 0.0001, 'PDOCO': 0.1, 'PDOMO': 0.01, 'non_pri': 0.001, 'non_comm': 0.0001, 'PDOMS': 0.01}

    if data_source == 'kddcup99' and num_of_clients == 8:
        eps_list = [10., 1., 0.1]
        alpha_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        stepsize = 4000
        recur_list = [i * stepsize * num_of_clients for i in range(1, 6)]
        rpt_times = 20
        is_minus_one = True
        for setting in setting_list:
            if 'adv' in setting:
                data = "../data/" + setting + "/kddcup99_processed_x_8.npy"
                target = "../data/" + setting + "/kddcup99_processed_y_8.npy"
            else:
                data = "../data/" + setting + "/kddcup99_processed_x_stoc_8.npy"
                target = "../data/" + setting + "/kddcup99_processed_y_stoc_8.npy"
            filename = "../" + setting + "/conf.ini"
            conf_save(data_source, data, target, eps_list=eps_list, recur_list=recur_list, rpt_times=rpt_times,
                      number_of_clients=num_of_clients, topology=topology, A=A, mu=mu, alpha_list=alpha_list,
                      is_minus_one=is_minus_one, alpha_dict=alpha_dict, filename=filename)
    elif data_source == 'kddcup99' and num_of_clients == 64:
        eps_list = [1.0, ]
        alpha_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        stepsize = 4000
        recur_list = [i * stepsize * num_of_clients for i in range(1, 6)]
        rpt_times = 20
        is_minus_one = True
        for setting in setting_list:
            if 'adv' in setting:
                data = "../data/" + setting + "/kddcup99_processed_x_64.npy"
                target = "../data/" + setting + "/kddcup99_processed_y_64.npy"
            else:
                data = "../data/" + setting + "/kddcup99_processed_x_stoc.npy"
                target = "../data/" + setting + "/kddcup99_processed_y_stoc.npy"
            filename = "../" + setting + "/conf.ini"
            conf_save(data_source, data, target, eps_list=eps_list, recur_list=recur_list, rpt_times=rpt_times,
                      number_of_clients=num_of_clients, topology=topology, A=A, mu=mu, alpha_list=alpha_list,
                      is_minus_one=is_minus_one, alpha_dict=alpha_dict, filename=filename)
    elif data_source == 'diabetes' and num_of_clients == 8:
        eps_list = [10., 1., 0.1]
        alpha_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        stepsize = 2500
        recur_list = [i * stepsize * num_of_clients for i in range(1, 6)]
        rpt_times = 20
        is_minus_one = True
        for setting in setting_list:
            if 'adv' in setting:
                data = "../data/" + setting + "/diabetes_processed_x.npy"
                target = "../data/" + setting + "/diabetes_processed_y.npy"
            else:
                data = "../data/" + setting + "/diabetes_processed_x_stoc.npy"
                target = "../data/" + setting + "/diabetes_processed_y_stoc.npy"
            filename = "../" + setting + "/conf.ini"
            conf_save(data_source, data, target, eps_list=eps_list, recur_list=recur_list, rpt_times=rpt_times,
                      number_of_clients=num_of_clients, topology=topology, A=A, mu=mu, alpha_list=alpha_list,
                      is_minus_one=is_minus_one, alpha_dict=alpha_dict, filename=filename)

    #######################################
    conf_dict = conf_load()
    data = conf_dict['data']
    target = conf_dict['target']
    eps_list = conf_dict['eps_list']
    recur_list = conf_dict['recur_list']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    # print(conf_dict)
    #######################################
