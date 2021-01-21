import os
import copy
import pickle
import datetime
import alipy
import shutil
import warnings
import autosklearn
import sklearn.datasets
from autosklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, minmax_scale
from extended_models.MLP import MLPClassifier
from algorithm_order import ActiveOrderCASH
from active_cash import get_alibox, fetch_dataset
from fix_candidate_alg import ALMS_alg

os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
warnings.simplefilter("ignore")


def ALMS_AL(dataset_id, arms_level, al_folds,
            test_ratio, ini_lab_ratio, data_home,
            al_save_dir, resampling_strategy,
            resampling_args, start_fold, end_fold, **kwargs):
    """AL main loop.

    :param dataset_name: str, used for a more fancy output.
    :param al_folds: int, number of folds.
    :param test_ratio: float, ratio of test data when performing data split.
    :param ini_lab_ratio: float, ratio of initially labeled data.
    :param bandit_trial_budget: int, budget of rising bandit trials in each iteration of AL.
        Note that, in each trial, each arm will be pulled once.
    :param smbo_time_limit_per_run: int, time limit (seconds) of each trial when performing smbo.
    :param query_budget: int, number of queries.
    :param al_save_dir: str, dir to save AL output files.
    :param cash_save_dir: str, dir to save cash output files.
    :param cash_tmp_dir: str, dir to save cash tmp files.
    :return:
    """
    # 'multinomial_nb', 'gaussian_nb', 'bernoulli_nb'
    if arms_level == 1:
        arms = ['gradient_boosting', ]
        # arms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    elif arms_level == 2:
        arms = ['passive_aggressive', 'k_nearest_neighbors', 'libsvm_svc', 'sgd', 'lda',
                'adaboost', 'random_forest', 'extra_trees', 'decision_tree', 'MLPClassifier']
    elif arms_level == 3:
        arms = ['adaboost', 'random_forest',
                'libsvm_svc', 'sgd',
                'extra_trees', 'decision_tree',
                'k_nearest_neighbors',
                'passive_aggressive', 'gradient_boosting',
                'lda', 'qda', 'MLPClassifier',
                ]
    else:
        raise ValueError("arms_level must in {1,2,3}.")

    dataset_name, X, y = fetch_dataset(dataset_id, data_home=data_home)

    if 10000 > len(X) > 3000:
        query_budget = 1000
    elif len(X) >= 10000:
        query_budget = 2000
    else:
        query_budget = 300

    alibox = get_alibox(al_save_dir, dataset_id, dataset_name,
                        X, y, test_ratio, ini_lab_ratio, al_folds,
                        resampling_strategy, resampling_args)
    results = []
    # disable the removal function for the first few queries

    for i in np.arange(start=start_fold, stop=end_fold):
        i = int(i)
        # clean_cash_dir(cash_save_dir=os.path.join(cash_save_dir, f"{dataset_id}_{dataset_name}_{i}"),
        #                cash_tmp_dir=os.path.join(cash_tmp_dir, f"{dataset_id}_{dataset_name}_{i}"))
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(i)
        arb = ALMS_alg(X=X, y=y, initial_labid=label_ind)
        # Get the data split of one fold experiment

        saver = alibox.get_stateio(i, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'ALMS_f{i}.pkl'))

        performance = arb.train_test(test_idx=test_idx)
        saver.set_initial_point(performance)
        print("initial point is ", performance)

        for j in range(min(query_budget, len(unlab_ind))):
            select_ind = arb.select(unlab_ind)
            print("query complete: ", select_ind)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
            # update model

            performance = arb.train_test(test_idx=test_idx)
            # save state
            st = alibox.State(select_index=select_ind, performance=performance)
            saver.add_state(st)
            saver.save()
            print(datetime.datetime.now())
        results.append(copy.deepcopy(saver))

    return results


def active_inas(dataset_id, arms_level, al_folds,
                test_ratio, ini_lab_ratio, data_home,
                bandit_trial_budget, ml_memory_limit,
                smbo_time_limit_per_run, parallel,
                query_budget, al_save_dir, tradeoff,
                cash_save_dir, cash_tmp_dir,
                resampling_strategy, resampling_args,
                alpha, start_fold, end_fold, process_limit, **kwargs):
    """AL main loop.

    :param dataset_name: str, used for a more fancy output.
    :param al_folds: int, number of folds.
    :param test_ratio: float, ratio of test data when performing data split.
    :param ini_lab_ratio: float, ratio of initially labeled data.
    :param bandit_trial_budget: int, budget of rising bandit trials in each iteration of AL.
        Note that, in each trial, each arm will be pulled once.
    :param smbo_time_limit_per_run: int, time limit (seconds) of each trial when performing smbo.
    :param query_budget: int, number of queries.
    :param al_save_dir: str, dir to save AL output files.
    :param cash_save_dir: str, dir to save cash output files.
    :param cash_tmp_dir: str, dir to save cash tmp files.
    :return:
    """

    arms = ['k_nearest_neighbors', 'lda', 'sgd',
            'passive_aggressive', 'decision_tree', 'extra_trees',
            'libsvm_svc', 'qda', 'MLPClassifier', 'random_forest',
            'adaboost', 'gradient_boosting'
            ]

    dataset_name, X, y = fetch_dataset(dataset_id, data_home=data_home)
    # # if large dataset double the search time
    if len(X) > 5000:
        smbo_time_limit_per_run *= 2
    if 10000 > len(X) > 3000:
        query_budget = 1000
    elif len(X) >= 10000:
        query_budget = 2000

    # os.makedirs(os.path.join(al_save_dir, dataset_id + '_' + dataset_name), exist_ok=True)
    alibox = get_alibox(al_save_dir, dataset_id, dataset_name,
                        X, y, test_ratio, ini_lab_ratio, al_folds,
                        resampling_strategy, resampling_args)
    results = []

    for i in np.arange(start=start_fold, stop=end_fold):
        i = int(i)
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(i)
        arb = ActiveOrderCASH(include_alg=arms, dataset_name=f"{dataset_id}_{dataset_name}_{i}",
                              trial_num=bandit_trial_budget,
                              metric=accuracy,
                              time_limit_per_run=smbo_time_limit_per_run,
                              output_dir=cash_save_dir,
                              tmp_dir=cash_tmp_dir,
                              time_limit=None, parallel=parallel,
                              n_jobs=None, ml_memory_limit=ml_memory_limit,
                              resampling_strategy=resampling_strategy,
                              resampling_args=resampling_args, alpha=alpha,
                              process_limit=process_limit, X=X, y=y, ava_idx=train_idx)
        arb.remove_arm_flag = False
        # Get the data split of one fold experiment
        # examine the existed files
        if os.path.exists(os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                       f'inas_f{i}_tr{tradeoff}.pkl')):
            saver = alipy.experiment.StateIO.load(os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'inas_f{i}.pkl'))
            saver._saving_dir = os.path.join(al_save_dir, dataset_id + '_' + dataset_name)
            queried_num = len(saver)
            if queried_num == query_budget or queried_num == len(unlab_ind):
                return

        saver = alibox.get_stateio(i, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'inas_f{i}_tr{tradeoff}.pkl'))

        performance = arb.search_and_train_test(X_train=X[label_ind].copy(), y_train=y[label_ind].copy(),
                                                X_test=X[test_idx].copy(), y_test=y[test_idx].copy(),
                                                lab_id=label_ind, enalbe_weighting=False)
        saver.set_initial_point(performance)
        print("initial point is ", performance)

        for j in range(min(query_budget, len(unlab_ind))):
            select_ind = arb._query_by_unc(X=X, y=y, lab_ind=label_ind, unlab_ind=unlab_ind, batch_size=1)
            print("query complete: ", select_ind)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
            # update model
            arb.refresh_cash()

            performance = arb.search_and_train_test(X_train=X[label_ind].copy(), y_train=y[label_ind].copy(),
                                                    X_test=X[test_idx].copy(), y_test=y[test_idx].copy(),
                                                    lab_id=label_ind, enalbe_weighting=False)
            # save state
            st = alibox.State(select_index=select_ind, performance=performance)
            st['best_config'] = arb.best_hpo_config
            st['curr_id'] = arb.current_arm_id
            saver.add_state(st)
            saver.save()
            print(datetime.datetime.now())

        results.append(copy.deepcopy(saver))
    return results
