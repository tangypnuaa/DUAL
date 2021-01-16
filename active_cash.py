import os
import copy
import pickle
import datetime
import alipy
import shutil
import autosklearn
import sklearn.datasets
from autosklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, minmax_scale
from extended_models.MLP import MLPClassifier
from algorithm import ActiveRisingBandit

__all__ = ['active_learning', 'fetch_dataset', 'random_cash_successive', 'random_cash']
autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def fetch_dataset(openml_id, data_home):
    dataset = sklearn.datasets.fetch_openml(data_id=str(openml_id), data_home=data_home, return_X_y=False)
    y_txt = dataset.target
    y = LabelEncoder().fit_transform(y=y_txt)
    X = np.asarray(dataset.data)
    print(
        f"Dataset: {dataset.details['name']}\nInstances: {len(y)}\nclasses: {len(np.unique(y))}\nfeatures: {dataset.data.shape[1]}")
    return dataset.details['name'], minmax_scale(X), y


def check_split(alibox_instance, folds, y, threshold):
    """check whether there are at least threshold examples in each class."""
    lab_space = np.unique(y)
    for i in range(folds):
        train_idx, test_idx, label_ind, unlab_ind = alibox_instance.get_split(i)
        lab_arr = y[label_ind]
        for lab in lab_space:
            if np.sum(lab_arr == lab) < threshold:
                return False
    return True


def clean_cash_dir(cash_save_dir, cash_tmp_dir):
    #  check if the tmp dir is exists
    if cash_save_dir is not None and os.path.exists(cash_save_dir):
        shutil.rmtree(cash_save_dir)
    # if os.path.exists(os.path.join(cash_tmp_dir, dataset_id + '_' + dataset_name)):
    #     shutil.rmtree(os.path.join(cash_tmp_dir, dataset_id + '_' + dataset_name))
    if cash_tmp_dir is not None and os.path.exists(cash_tmp_dir):
        shutil.rmtree(cash_tmp_dir)


def get_alibox(al_save_dir, dataset_id, dataset_name,
               X, y, test_ratio, ini_lab_ratio, al_folds,
               resampling_strategy, resampling_args,
               rcash=False, rcash_save_dir=None):
    one_fold_flag = False
    split_flag = False
    # read data split
    if os.path.exists(os.path.join(al_save_dir, dataset_id + '_' + dataset_name, 'al_settings.pkl')):
        try:
            alibox = alipy.ToolBox.load(os.path.join(al_save_dir, dataset_id + '_' + dataset_name, 'al_settings.pkl'))
            alibox._saving_path = os.path.join(al_save_dir, dataset_id + '_' + dataset_name)
            alibox._saving_dir = os.path.join(al_save_dir, dataset_id + '_' + dataset_name)
            train_idxs, test_idxs, label_inds, unlab_inds = alibox.get_split()
        except:
            alibox = alipy.ToolBox(X=X, y=y, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name))
            alibox._saving_path = os.path.join(al_save_dir, dataset_id + '_' + dataset_name)
            alibox._saving_dir = os.path.join(al_save_dir, dataset_id + '_' + dataset_name)
            train_idxs, test_idxs, label_inds, unlab_inds = alipy.data_manipulate.split_load(
                os.path.join(al_save_dir, dataset_id + '_' + dataset_name))
            alibox.train_idx = train_idxs
            alibox.test_idx = test_idxs
            alibox.label_idx = label_inds
            alibox.unlabel_idx = unlab_inds
        if len(np.shape(train_idxs)) == 1:  # only 1 fold is performed
            one_fold_flag = True
            split_flag = True
            print(f"Only one fold split is found for dataset {dataset_id}")
    else:
        if rcash is False:
            alibox = alipy.ToolBox(X=X, y=y, query_type='AllLabels',
                                   saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name))
        else:
            assert rcash_save_dir is not None
            alibox = alipy.ToolBox(X=X, y=y, query_type='AllLabels',
                                   saving_path=os.path.join(rcash_save_dir, dataset_id + '_' + dataset_name))
        start_fold = 0
        split_flag = True

    if split_flag:
        # Split data with fixed random state
        print("splitting data...")
        alibox.split_AL(test_ratio=test_ratio, initial_label_rate=ini_lab_ratio, split_count=al_folds,
                        all_class=True)
        if resampling_strategy == 'cv':
            split_num = 0
            while not check_split(alibox, folds=al_folds, y=y, threshold=resampling_args):
                alibox.split_AL(test_ratio=test_ratio, initial_label_rate=ini_lab_ratio, split_count=al_folds,
                                all_class=True)
                split_num += 1
                if split_num % 100 == 0:
                    print(f"split has been tried {split_num}+ times.\r", end='')
                if split_num > 50000:
                    print("The dataset is extremely imbalance. Try to reduce the cv folds or use holdout instead.")
                    exit()
            print("data split ok.")
        alibox.save()

    if one_fold_flag:
        alibox.train_idx[0] = train_idxs
        alibox.test_idx[0] = test_idxs
        alibox.label_idx[0] = label_inds
        alibox.unlabel_idx[0] = unlab_inds
        alibox.save()

    alibox._split = True
    alibox.split_count = al_folds
    return alibox


def active_learning(dataset_id, arms_level, al_folds,
                    test_ratio, ini_lab_ratio, data_home,
                    bandit_trial_budget, ml_memory_limit,
                    smbo_time_limit_per_run, parallel,
                    query_budget, al_save_dir, tradeoff,
                    cash_save_dir, cash_tmp_dir,
                    resampling_strategy, resampling_args,
                    enable_removing_iter,
                    alpha, start_fold, end_fold, process_limit, enable_weighting=True):
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
    # # if large dataset double the search time
    if len(X) > 5000:
        smbo_time_limit_per_run *= 2

    os.makedirs(os.path.join(al_save_dir, dataset_id + '_' + dataset_name), exist_ok=True)
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
        arb = ActiveRisingBandit(include_alg=arms, dataset_name=f"{dataset_id}_{dataset_name}_{i}",
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

        saver = alibox.get_stateio(i, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'DUAL_f{i}_tr{tradeoff}.pkl'))

        performance = arb.search_and_train_test(X_train=X[label_ind].copy(), y_train=y[label_ind].copy(),
                                                X_test=X[test_idx].copy(), y_test=y[test_idx].copy(),
                                                lab_id=label_ind)
        saver.set_initial_point(performance)
        print("initial point is ", performance)
        # save the best initial model for comparing different AL strategies
        with open(os.path.join(al_save_dir, dataset_id + '_' + dataset_name, f'im_{dataset_name}_{i}_t{tradeoff}.pkl'),
                  'wb') as f:
            pickle.dump(arb.best_model, f)

        refresh_count = 1
        for j in range(min(query_budget, len(unlab_ind))):
            select_ind = arb.query_by_EE(X, y, label_ind, unlab_ind, batch_size=1,
                                         trade_off=tradeoff if tradeoff > 0 else len(arb.arm_candidate) / 10)
            print("query complete: ", select_ind)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
            # update model
            arb.refresh_cash()

            performance = arb.search_and_train_test(X_train=X[label_ind].copy(), y_train=y[label_ind].copy(),
                                                    X_test=X[test_idx].copy(), y_test=y[test_idx].copy(),
                                                    lab_id=label_ind)
            # save state
            st = alibox.State(select_index=select_ind, performance=performance)
            st['arms_cand'] = arb.arm_candidate
            st['best_config'] = arb.best_hpo_config
            st['rewards'] = arb.rewards
            saver.add_state(st)
            saver.save()
            print(datetime.datetime.now())

            # start to remove arms after x queries
            if j > enable_removing_iter:
                # print("start to remove arms")
                arb.remove_arm_flag = True

        results.append(copy.deepcopy(saver))

    return results


def random_cash(dataset_id, arms_level, al_folds,
                test_ratio, ini_lab_ratio, data_home,
                bandit_trial_budget, ml_memory_limit,
                smbo_time_limit_per_run, parallel,
                query_budget, al_save_dir, tradeoff,
                cash_save_dir, cash_tmp_dir,
                resampling_strategy, resampling_args,
                enable_removing_iter,
                alpha, start_fold, end_fold, process_limit):
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
    # # if large dataset double the search time
    if len(X) > 5000:
        smbo_time_limit_per_run *= 2

    os.makedirs(os.path.join(al_save_dir, dataset_id + '_' + dataset_name), exist_ok=True)
    alibox = get_alibox(al_save_dir, dataset_id, dataset_name,
                        X, y, test_ratio, ini_lab_ratio, al_folds,
                        resampling_strategy, resampling_args)
    results = []
    # disable the removal function for the first few queries

    for i in np.arange(start=start_fold, stop=end_fold):
        i = int(i)
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(i)
        arb = ActiveRisingBandit(include_alg=arms, dataset_name=f"{dataset_id}_{dataset_name}_{i}",
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

        saver = alibox.get_stateio(i, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'random_f{i}.pkl'))

        q_num = 0
        for j in range(min(query_budget, len(unlab_ind))):
            if len(unlab_ind) == 0:
                break
            if q_num >= query_budget:
                break
            btsz = 5
            if len(unlab_ind) <= btsz:
                select_ind = unlab_ind
            else:
                select_ind = np.random.choice(unlab_ind, size=5, replace=False)
            q_num += len(select_ind)
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
            # st['arms_cand'] = arb.arm_candidate
            st['best_config'] = arb.best_hpo_config
            # st['rewards'] = arb.rewards
            saver.add_state(st)
            saver.save()
            print(datetime.datetime.now())

        results.append(copy.deepcopy(saver))
    return results


def random_cash_successive(dataset_id, arms_level, al_folds,
                           test_ratio, ini_lab_ratio, data_home,
                           bandit_trial_budget, ml_memory_limit,
                           smbo_time_limit_per_run, parallel,
                           query_budget, al_save_dir, tradeoff,
                           cash_save_dir, cash_tmp_dir,
                           resampling_strategy, resampling_args,
                           enable_removing_iter,
                           alpha, start_fold, end_fold, process_limit):
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
    # # if large dataset double the search time
    if len(X) > 5000:
        smbo_time_limit_per_run *= 2

    os.makedirs(os.path.join(al_save_dir, dataset_id + '_' + dataset_name), exist_ok=True)
    alibox = get_alibox(al_save_dir, dataset_id, dataset_name,
                        X, y, test_ratio, ini_lab_ratio, al_folds,
                        resampling_strategy, resampling_args)
    results = []
    # disable the removal function for the first few queries

    for i in np.arange(start=start_fold, stop=end_fold):
        i = int(i)
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(i)
        arb = ActiveRisingBandit(include_alg=arms, dataset_name=f"{dataset_id}_{dataset_name}_{i}",
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

        saver = alibox.get_stateio(i, saving_path=os.path.join(al_save_dir, dataset_id + '_' + dataset_name,
                                                               f'random_f{i}_rising.pkl'))

        for j in range(min(query_budget, len(unlab_ind))):
            select_ind = np.random.choice(unlab_ind, size=1, replace=False)
            print("query complete: ", select_ind)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
            # update model
            arb.refresh_cash()

            performance = arb.search_and_train_test(X_train=X[label_ind].copy(), y_train=y[label_ind].copy(),
                                                    X_test=X[test_idx].copy(), y_test=y[test_idx].copy(),
                                                    lab_id=label_ind, enalbe_weighting=True)
            # save state
            st = alibox.State(select_index=select_ind, performance=performance)
            st['arms_cand'] = arb.arm_candidate
            st['best_config'] = arb.best_hpo_config
            st['rewards'] = arb.rewards
            saver.add_state(st)
            saver.save()
            print(datetime.datetime.now())

            # # start to remove arms after x queries
            if j > enable_removing_iter:
                # print("start to remove arms")
                arb.remove_arm_flag = True

        results.append(copy.deepcopy(saver))
    return results

