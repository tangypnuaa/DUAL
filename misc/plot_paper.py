import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import operator
from active_cash import fetch_dataset

tmp_home = "/path/to/save_home"
al_home = tmp_home + "al_out"
data_home = tmp_home + "openml"
datasets = [54, 50, 1501, 36, 40670, 3, 40701, 1489, 28, 4534, 1046, 6]
methods = ["DUAL", "QueryInstanceCoresetGreedy",
           "QueryInstanceUncertainty", "QueryInstanceQUIRE",
           "QueryInstanceRandom", "random_CASH", "inas", "ALMS"]


def plot_lc(dataset_id, method_arr: 'list',
            isSmooth=True, legend=False, use_accurate_cost=False, budget_array=None,
            smooth_r=2, smooth_st=0, fold=10):
    global data_home, al_home
    assert smooth_st < smooth_r
    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=8)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    dataset_name, X, y = fetch_dataset(dataset_id, data_home=data_home)

    query_budget = 300
    if 10000 > len(X) > 3000:
        query_budget = 1000
    elif len(X) > 10000:
        query_budget = 1500
    else:
        query_budget = 300

    fig, ax = plt.subplots()
    ini_pt_arr = []
    for mth_id, mth in enumerate(method_arr):
        x_all = []
        y_all = []
        for fo in range(fold):
            if mth == "DUAL":
                saver = pickle.load(
                    open(os.path.join(al_home, str(dataset_id) + '_' + dataset_name, f'DUAL_f{fo}_tr-1.0.pkl'), 'rb'))
                ini_pt_arr.append(saver.initial_point)
            elif mth == "random_CASH":
                saver = pickle.load(
                    open(os.path.join(al_home, str(dataset_id) + '_' + dataset_name, f'random_f{fo}_rising.pkl'), 'rb'))
            else:
                saver = pickle.load(
                    open(os.path.join(al_home, str(dataset_id) + '_' + dataset_name, f'{mth}_f{fo}.pkl'), 'rb'))

            # assert mth in name_arr, f'un-known method {mth}'
            x, y = [0], [ini_pt_arr[fo]]
            cost_x = 0
            for it in range(query_budget):
                if isSmooth:
                    if (it + smooth_st) % smooth_r != 0 and it != 0:
                        cost_x += 1
                        continue
                mmap = saver[it]["performance"]
                cost_x += 1
                x.append(cost_x)
                y.append(mmap)
            x_all.append(x)
            y_all.append(y)

        msize = 6
        if 1000 >= query_budget > 300:
            msize = 4
        elif query_budget > 1000:
            msize = 3
        ax.plot(np.mean(x_all, axis=0), np.mean(y_all, axis=0), label=methods_label[mth_id],
                linewidth=methods_linewodth[mth_id],
                color=methods_color[mth_id],
                linestyle=methods_lstyle[mth_id],
                marker=methods_marker[mth_id],
                markersize=msize if 'DUAL' in mth else msize-2
                )
        if legend:
            ax.legend()
    plt.xlabel("number of queries")
    plt.ylabel("accuracy")
    # ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')

    fig.tight_layout()
    fig.savefig(f'lc_{dataset_name}.png', dpi=200, bbox_inches='tight')
    fig.show()


methods_label = ["DUAL", "Coreset", "Entropy", "QUIRE", "Random", "CASH", "Active-iNAS", "ALMS"]
methods_linewodth = [2.8, 1.7, 1.7, 1.7, 1.7, 1.7,
                     1.7, 1.7, 1.7, 1.3, 1.3, 1.3, 1.3, 1.7]
methods_lstyle = ['-', '--', '--',
                  '--', '--',
                  '--', '--', '--', '--', '--']
methods_color = ['#F71E35', '#274c5e', '#0080ff',
                 '#bf209f', '#79bd9a', 'gray', 'black', '#679b00', 'black']
methods_marker = ["D", "d", "^",
                  "^", "o", "^", "o", "^"]
for did, da in enumerate(datasets):
    plot_lc(dataset_id=da, method_arr=methods,
            isSmooth=True, legend=True, use_accurate_cost=False, budget_array=None,
            smooth_r=5, fold=10)
