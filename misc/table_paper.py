import os
import copy
import numpy as np
import pickle
from alipy.utils.interface import BaseAnalyser
from sklearn.datasets import fetch_openml

tmp_home = "/path/to/save_home"
al_home = tmp_home + "al_out"
data_home = tmp_home + "openml"
datasets = [54, 50, 1501, 36, 40670, 3, 40701, 1489, 28, 4534, 1046, 6]
methods = ["DUAL", "random_CASH", "QueryInstanceCoresetGreedy",
           "QueryInstanceUncertainty", "QueryInstanceRandom", "QueryInstanceQUIRE"]


def extract_mat(saver_arr, extract_keys, max_len=300):
    extracted_matrix = []
    for stateio in saver_arr:
        stateio_line = []
        for state in stateio:
            if extract_keys not in state.keys():
                raise ValueError('The extract_keys should be a subset of the keys of each State object.\n'
                                 'But keys in the state are: %s' % str(state.keys()))
            if extract_keys == "arms_cand":
                stateio_line.append(len(state.get_value(extract_keys)))
            else:
                stateio_line.append(state.get_value(extract_keys))

            if len(stateio_line) >= max_len:
                break
        extracted_matrix.append(copy.copy(stateio_line))
    return extracted_matrix


def sampling(mat, every=5):
    return np.asarray(mat)[:, np.arange(every - 1, len(mat[0]), every)]


def get_mean_curve_value(openml_id, extract_key, ttest_max=True, max_iter=300, sampling_flag=True, get_random=True):
    dataset = fetch_openml(data_id=str(openml_id), data_home=tmp_home + 'openml', return_X_y=False)
    ins_num = len(dataset.target)
    fea_num = dataset.data.shape[1]
    cla_num = len(np.unique(dataset.target))
    data_name = dataset.details['name']
    folds = range(10)

    mean_arr = []
    std_arr = []
    ttest_arr = []
    ttest_res = [0] * 7

    # our method
    for tradeoff in [0.1, 1.0, 10.0, 0.0, -1.0]:
        saver_arr = []
        for fo in folds:
            saver = pickle.load(
                open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'DUAL_f{fo}_tr{tradeoff}.pkl'), 'rb'))
            # saver.recover_workspace(300)
            saver_arr.append(saver)
        mat = np.asarray(extract_mat(saver_arr, extract_key))
        mat = mat[:, :max_iter]
        if sampling_flag:
            mat = sampling(mat)
        mat = np.mean(mat, axis=1)
        ttest_arr.append(mat)
        mean_arr.append(np.mean(mat))
        std_arr.append(np.std(mat))

    # random cash
    saver_arr = []
    for fo in folds:
        saver = pickle.load(
            open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'random_f{fo}_rising.pkl'), 'rb'))
        # saver.recover_workspace(300)
        saver_arr.append(saver)
    mat = np.asarray(extract_mat(saver_arr, extract_key))
    mat = mat[:, :max_iter]
    if sampling_flag:
        mat = sampling(mat)
    mat = np.mean(mat, axis=1)
    ttest_arr.append(mat)
    mean_arr.append(np.mean(mat))
    std_arr.append(np.std(mat))

    # random
    saver_arr = []
    for fo in folds:
        saver = pickle.load(
            open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'random_f{fo}.pkl'), 'rb'))
        # saver.recover_workspace(300)
        saver_arr.append(saver)
    if extract_key == "arms_cand":
        mean_arr.append(12)
        std_arr.append(0)
    else:
        mat = extract_mat(saver_arr, extract_key)
        mat = np.mean(mat, axis=1)
        ttest_arr.append(mat)
        mean_arr.append(np.mean(mat))
        std_arr.append(np.std(mat))

    # ttest
    if ttest_max:
        mid = np.argmax(mean_arr)
    else:
        mid = np.argmin(mean_arr)
    ttest_res[mid] = 1
    for iarr, array in enumerate(ttest_arr):
        if iarr != mid:
            ttest_one = BaseAnalyser.paired_ttest(ttest_arr[mid], array)
            ttest_res[iarr] = 1 - ttest_one
    if len(np.unique(ttest_res)) == 1:
        ttest_res = [0] * 7

    return data_name, mean_arr, std_arr, ttest_res


def get_abl(openml_id, extract_key, ttest_max=True, max_iter=300, sampling_flag=True, get_random=True):
    dataset = fetch_openml(data_id=str(openml_id), data_home=tmp_home + 'openml', return_X_y=False)
    ins_num = len(dataset.target)
    fea_num = dataset.data.shape[1]
    cla_num = len(np.unique(dataset.target))
    data_name = dataset.details['name']
    folds = range(10)

    mean_arr = []
    std_arr = []
    ttest_arr = []
    ttest_res = [0] * 3

    # our method
    for tradeoff in [-1.0]:
        saver_arr = []
        for fo in folds:
            saver = pickle.load(
                open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'EE_f{fo}_tr{tradeoff}.pkl'), 'rb'))
            # saver.recover_workspace(300)
            saver_arr.append(saver)
        mat = np.asarray(extract_mat(saver_arr, extract_key, max_iter))
        # mat = mat[:, :max_iter]
        if sampling_flag:
            mat = sampling(mat, every=5)
        mat = np.mean(mat, axis=1)
        ttest_arr.append(mat)
        mean_arr.append(np.mean(mat))
        std_arr.append(np.std(mat))

    # random cash
    saver_arr = []
    for fo in folds:
        saver = pickle.load(
            open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'random_f{fo}_rising.pkl'), 'rb'))
        # saver.recover_workspace(300)
        saver_arr.append(saver)
    mat = np.asarray(extract_mat(saver_arr, extract_key, max_iter))
    # mat = mat[:, :max_iter]
    if sampling_flag:
        mat = sampling(mat, every=5)
    mat = np.mean(mat, axis=1)
    ttest_arr.append(mat)
    mean_arr.append(np.mean(mat))
    std_arr.append(np.std(mat))

    # random
    saver_arr = []
    for fo in folds:
        saver = pickle.load(
            open(os.path.join(al_home, str(openml_id) + '_' + data_name, f'random_f{fo}.pkl'), 'rb'))
        # saver.recover_workspace(300)
        saver_arr.append(saver)
    if extract_key == "arms_cand":
        mean_arr.append(12)
        std_arr.append(0)
        ttest_arr.append([12]*10)
    else:
        mat = extract_mat(saver_arr, extract_key)
        mat = np.mean(mat, axis=1)
        ttest_arr.append(mat)
        mean_arr.append(np.mean(mat))
        std_arr.append(np.std(mat))

    # ttest
    if ttest_max:
        mid = np.argmax(mean_arr)
    else:
        mid = np.argmin(mean_arr)
    ttest_res[mid] = 1
    for iarr, array in enumerate(ttest_arr):
        if iarr != mid:
            ttest_one = BaseAnalyser.paired_ttest(ttest_arr[mid], array)
            ttest_res[iarr] = 1 - ttest_one
    if len(np.unique(ttest_res)) == 1:
        ttest_res = [0] * 3

    return data_name, mean_arr, std_arr, ttest_res


def to_latex_code(dataset_name, mean_arr, std_arr, ttest_res, type="acc"):
    line = f"{dataset_name} "
    for it, item in enumerate(mean_arr):
        if type == "acc":
            if ttest_res[it] == 1:
                line += f"& $\\bm{'{'}{item * 100:.1f}{'}'}\\pm \\bm{'{'}{std_arr[it] * 100:.1f}{'}'}$ "
            else:
                line += f"& ${item * 100:.1f}\\pm {std_arr[it] * 100:.1f}$ "
        else:
            if ttest_res[it] == 1:
                line += f"& $\\bm{'{'}{item:.1f}{'}'}\\pm \\bm{'{'}{std_arr[it]:.1f}{'}'}$ "
            else:
                line += f"& ${item:.1f}\\pm {std_arr[it]:.1f}$ "
    line += "\\\\"
    return line


latex_lines = []
for ek in ["performance", "arms_cand"]:
    for ddid, did in enumerate(datasets):
        data_name, mean_arr, std_arr, ttest_arr = \
            get_abl(openml_id=did, extract_key=ek,
                     ttest_max=True if ek == "performance" else False,
                     max_iter=300 if ek == "performance" else 300,
                     sampling_flag=True if ek == "performance" else True)
        # print(mean_arr, std_arr)
        ll = to_latex_code(dataset_name=data_name, mean_arr=mean_arr, std_arr=std_arr,
                           ttest_res=ttest_arr,
                           type="acc" if ek == "performance" else "ff")
        if ek == "performance":
            latex_lines.append(ll)
        else:
            latex_lines[ddid] += ll
        print(ll)
        # print("\\hline")
    print(os.linesep)

for lll in latex_lines:
    print(lll)

