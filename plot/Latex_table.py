import json
import numpy as np


def table_load(filename='class_err_ret'):
    with open(filename) as fd:
        tab_dict = json.load(fd)
    return tab_dict


if __name__ == '__main__':
    tab_dict = table_load()
    source, recursion = 'diabetes', 100000
    # source, recursion = 'kddcup99', 160000
    epsilon_list = [0.1, 1.0, 10.0]
    alpha_list = [0.0, 0.1, 0.01, 0.001, 0.0001]
    topology = 'cycle_8'
    alg_list = ['DOCO', 'PDOCO', 'PDOMO', 'non_pri', 'PDOMS', 'PFTAL']
    alg_list = [alg + '_' + repr(recursion) for alg in alg_list]
    print(alg_list)

    '''
    alg_array = []
    for alg in alg_list:
        for alpha in alpha_list:
            if (alg.startswith('DOCO') or alg.startswith('non_pri')):
                alg_array.append(alg + '_' + repr(alpha))
            else:
                for epsilon in epsilon_list:
                    alg_array.append(alg + '_' + repr(alpha) + '_' + repr(epsilon))
    '''

    # len_table = np.floor(len(alg_array) / 3)
    len_table = len(alg_list)

    table_store = []
    max_ind_store = []
    alg_name_store = []

    for alg in alg_list:
        if (alg.startswith('DOCO') or alg.startswith('non_pri')):
            alg_name_store.append(alg)
            table_line = []
            index_store = 0
            err_rate_store = 1
            cnt = 0
            for alpha in alpha_list:
                alg_instance = alg + '_' + repr(alpha)
                err_rate = np.around(tab_dict[alg_instance], decimals=3)
                table_line.append(err_rate)
                if(cnt > 0 and err_rate < err_rate_store):
                    index_store = cnt
                    err_rate_store = err_rate
                cnt += 1
            table_store.append(table_line)
            max_ind_store.append(index_store)

    for epsilon in epsilon_list:
        for alg in alg_list:
            if (not alg.startswith('DOCO') and not alg.startswith('non_pri')):
                alg_name_store.append(alg)
                table_line = []
                index_store = 0
                err_rate_store = 1
                cnt = 0
                for alpha in alpha_list:
                    alg_instance = alg + '_' + repr(alpha) + '_' + repr(epsilon)
                    err_rate = np.around(tab_dict[alg_instance], decimals=3)
                    table_line.append(err_rate)
                    if (cnt > 0 and err_rate < err_rate_store):
                        index_store = cnt
                        err_rate_store = err_rate
                    cnt += 1
                table_store.append(table_line)
                max_ind_store.append(index_store)

    print(table_store)
    print(max_ind_store)

    table_str = ''
    for i in range(len(max_ind_store)):
        cnt = 0
        table_str += alg_name_store[i][:-7]
        for alpha in alpha_list:
            table_str += '& '
            if (cnt == max_ind_store[i]):
                table_str += '\\textbf{' + repr(table_store[i][cnt]) + '}'
            else:
                table_str += repr(table_store[i][cnt])
            cnt += 1
        table_str += '\\\\\n'

    with open('acc_table', 'w') as fd:
        fd.write(table_str)

    # cnt = 0
    # cnt_row = 0
    # tmp_ind_1 = [1, 1, 1]
    # tmp_ind_2 = [6, 6, 6]
    # tmp_1 = 0
    # tmp_2 = 0

    '''
    for i in alg_array:
        acc_tmp = float(tab_dict[i]) * 100
        acc_tmp = np.around(acc_tmp, 1)
        # print(cnt)
        # print(cnt_row)
        # print()
        if (cnt >= 1 and cnt < 5 and acc_tmp > tmp_1):
            # print(cnt)
            tmp_1 = acc_tmp

            tmp_ind_1[cnt_row] = cnt
        if (cnt >= 6 and acc_tmp > tmp_2):
            tmp_2 = acc_tmp
            tmp_ind_2[cnt_row] = cnt

        if (cnt == len_table - 1):
            tmp_1 = 0
            tmp_2 = 0
            cnt = -1
            cnt_row += 1
        cnt += 1

    # print(tmp_ind_1)
    # print(tmp_ind_2)

    cnt = 0
    cnt_row = 0
    for i in alg_array:
        acc_tmp = float(tab_dict[i]) * 100
        acc_tmp = np.around(acc_tmp, 1)

        
        if (cnt == tmp_ind_1[cnt_row] or cnt == tmp_ind_2[cnt_row]):
            table_str += '\\textbf{' + repr(acc_tmp) + '}'
        else:
            table_str += repr(acc_tmp)
            
        cnt += 1
        if (cnt < len_table):
            table_str += ' & '
        else:
            table_str += '\n& '
            cnt = 0
            cnt_row += 1

    with open('acc_table', 'w') as fd:
        fd.write(table_str)
    '''
