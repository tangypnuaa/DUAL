import copy
import logging
import os
import sys
import time
import warnings
import alipy
import pickle
import random
import shutil
from typing import List
import multiprocessing
import dask
import re
import dask.distributed
from inspect import signature
from concurrent.futures import ProcessPoolExecutor

# dask.config.set({'distributed.worker.daemon': False})

import autosklearn.classification
import numpy as np
from utils.my_scorer import weighted_acc
from alipy.utils.misc import randperm
from autosklearn.metrics import accuracy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from autosklearn.metrics import make_scorer
from sklearn.neural_network import MLPClassifier as skmlp

__all__ = ['ActiveOrderCASH']

name2name = {
    'adaboost': 'AdaboostClassifier',
    'bernoulli_nb': 'BernoulliNB',
    'decision_tree': 'DecisionTree',
    'extra_trees': 'ExtraTreesClassifier',
    'gradient_boosting': 'GradientBoostingClassifier',
    'k_nearest_neighbors': 'KNearestNeighborsClassifier',
    'lda': 'LDA',
    'liblinear_svc': 'LibLinear_SVC',
    'libsvm_svc': 'LibSVM_SVC',
    'passive_aggressive': 'PassiveAggressive',
    'qda': 'QDA',
    'random_forest': 'RandomForest',
    'sgd': 'SGD',
    "MLPClassifier": "MLP"
}


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def nlargestarg(a, n):
    """Return n largest values' indexes of the given array a.
    Parameters
    ----------
    a: {list, np.ndarray}
        Data array.
    n: int
        The number of returned args.
    Returns
    -------
    nlargestarg: list
        The n largest args in array a.
    """
    assert (n > 0)
    argret = np.argsort(a)
    # ascent
    return argret[argret.size - n:]


def update_seed():
    return random.randint(1, 2 ** 31)


def check_val_split(val_id, train_id, lab):
    all_lab = np.unique(lab)
    if len(np.unique(lab[val_id])) != len(all_lab):
        return False
    if len(np.unique(lab[train_id])) != len(all_lab):
        return False
    return True


class ActiveOrderCASH(object):
    def __init__(self, trial_num,
                 include_alg: List[str],
                 metric=None,
                 time_limit_per_run=300,
                 output_dir="logs",
                 tmp_dir="tmp",
                 dataset_name='default_dataset',
                 time_limit=None,
                 n_jobs=None,
                 ml_memory_limit=2048,
                 parallel=True,
                 alpha=10,
                 resampling_strategy="cv",
                 resampling_args=3,
                 process_limit=False,
                 X=None, y=None,
                 ava_idx=None):
        """
        :param metric: callable or None, a callable metric function in sklearn
        :param trial_num: int, total budget of time units for rising bandit.
        :param classifier_ids: subset of {'adaboost','bernoulli_nb','decision_tree','extra_trees','gaussian_nb','gradient_boosting',
        'gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','multinomial_nb','passive_aggressive','qda',
        'random_forest','sgd'}
        """
        self.timestamp = time.time()
        self.X = X
        self.y = y
        self.ava_idx = np.asarray(ava_idx)
        # self.trial_num = trial_num
        self.n_jobs = 1
        self.metric = metric
        self.alpha = alpha
        self.B = 0.01
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.time_limit_per_run = time_limit_per_run
        self.conf_limit = 2
        self.time_limit = time_limit
        self.trial_num = trial_num
        self.ml_memory_limit = ml_memory_limit
        self.parallel = parallel

        # Best configuration.
        self.optimal_algo_id = None
        self.optimal_configuration = None
        self.nbest_algo_ids = None
        self.best_lower_bounds = None
        self.es = None

        # Set up backend.
        self.dataset_name = dataset_name
        self.start_time = time.time()
        self.logger = logging.getLogger("rb")
        self.logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(logging.StreamHandler())
        self.query_times = 0
        self.fix_query_times = 0
        self.is_sharing = True
        self.process_limit = process_limit

        # Bandit settings.
        self.remove_arm_flag = True  # only true will enble the arm removing
        self.incumbent_perf = -float("INF")
        self.arms = include_alg
        self.include_algorithms = include_alg
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()
        self.fe_datanodes = dict()
        self.sub_best_model = dict()
        self.sub_best_reward = dict()
        for arm in self.arms:
            self.rewards[arm] = list()
            self.sub_best_model[arm] = None
            self.sub_best_reward[arm] = 0
            self.evaluation_cost[arm] = list()
            self.fe_datanodes[arm] = list()

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()
        self.seed = update_seed()

        # rising bandit
        # Initialize the parameters.
        self.arm_num = len(self.arms)
        self.arm_candidate = self.arms.copy()
        self.best_model = None
        self.best_hpo_config = None
        self.best_perf = 0
        self.first_play = True

        # order
        self.current_arm = self.arms[0]
        self.current_arm_id = 0

    def _train_domain_D(self, queried_id, max_iter: int = 500, num_cv=1):
        # split
        cash_valid_list = []
        cash_train_list = []
        domain_D_list = []
        valid_weights_list = []
        for s in range(num_cv):
            while True:
                cash_valid_idx = queried_id.random_sampling(rate=min(0.367, 1.0 / num_cv))
                cash_train_idx = copy.deepcopy(queried_id).difference_update(cash_valid_idx)
                if check_val_split(cash_valid_idx, cash_train_idx, self.y):
                    break
            cash_valid_list.append(cash_valid_idx)
            cash_train_list.append(cash_train_idx)

            # weigh the valid data
            domain_D = skmlp(hidden_layer_sizes=10, alpha=0, solver='sgd',
                             learning_rate='adaptive')
            # train D. Let t_valid = 0, t_all = 1, the importance weight is : (1-D)/D
            d_q = self.X[cash_train_idx]
            batch_size = len(cash_train_idx)
            y_D = [0] * batch_size + [1] * batch_size
            for sgd_epoch in range(max_iter):
                d_d = np.random.choice(self.ava_idx, len(cash_train_idx), replace=False)
                d_d = self.X[d_d]
                domain_D.partial_fit(X=np.concatenate((d_q, d_d), axis=0), y=y_D, classes=[0, 1])
            domain_D_list.append(domain_D)

            valid_pred = domain_D.predict_proba(self.X[cash_valid_idx])
            D_vablid_out = valid_pred[:, 1]
            valid_weights = (1 - D_vablid_out) / D_vablid_out
            valid_weights[valid_weights > np.sqrt(batch_size)] = np.sqrt(batch_size)
            valid_weights_list.append(valid_weights)

        return cash_valid_list, valid_weights_list

    def refresh_cash(self):

        self.fix_query_times = self.query_times
        self.is_sharing = False
        # self.best_model = None
        self.best_hpo_config = None
        self.best_perf = 0
        for arm in self.arms:
            self.sub_best_model[arm] = None
            self.sub_best_reward[arm] = 0

    def play_arm(self, X_train, y_train, arm, val_id, val_weights):
        logging.disable(sys.maxsize)
        automl = autosklearn.classification.AutoSklearnClassifier(
            include_estimators=[arm],
            include_preprocessors=["no_preprocessing", ],
            exclude_preprocessors=None,
            ensemble_size=0,
            initial_configurations_via_metalearning=0,
            memory_limit=self.ml_memory_limit,
            n_jobs=self.n_jobs, per_run_time_limit=self.time_limit_per_run,
            time_left_for_this_task=self.time_limit_per_run, seed=self.seed,
            tmp_folder=os.path.join(self.tmp_dir, self.dataset_name, str(self.seed), arm),
            delete_tmp_folder_after_terminate=True,
            output_folder=os.path.join(self.output_dir, self.dataset_name, str(self.seed), arm),
            delete_output_folder_after_terminate=True,
            resampling_strategy=PredefinedSplit,
            resampling_strategy_arguments={'test_fold': val_id},
            metric=make_scorer(name="weighted_acc", score_func=weighted_acc(val_weights))
        )
        self.seed = update_seed()
        automl.fit(X_train, y_train)
        # automl.fit_ensemble(y_train, ensemble_size=1)
        logging.disable(logging.NOTSET)
        best_try_idx = np.argmax(automl.cv_results_['mean_test_score'])
        performance = automl.cv_results_['mean_test_score'][best_try_idx]
        best_config = automl.cv_results_['params'][best_try_idx]

        global name2name
        prefix = 'classifier:'
        args_dict = dict()
        model_name = best_config['classifier:__choice__']
        for k in best_config.keys():
            if k.startswith(f"{prefix}{model_name}:"):
                arg_name = re.sub(f"{prefix}{model_name}:*", "", k)
                if best_config[k] == 'False':
                    args_dict[arg_name] = False
                elif best_config[k] == 'True':
                    args_dict[arg_name] = True
                else:
                    args_dict[arg_name] = best_config[k]
        if model_name == "MLPClassifier":
            from extended_models.MLP import MLPClassifier
            model_instance = MLPClassifier(**args_dict)
        else:
            exec(f"from autosklearn.pipeline.components.classification.{model_name} import {name2name[model_name]}")
            model_instance = eval(f"{name2name[model_name]}(**args_dict)")
        del automl
        # if model_instance is None:
        #     print('')
        return performance, best_config, model_instance

    @staticmethod
    def _play_arm_parallel_wrapper(ml_memory_limit, time_limit_per_run, tmp_dir, dataset_name,
                                   fix_query_times, output_dir, resampling_strategy, resampling_args,
                                   is_sharing, metric, seed, X_train, y_train, arm, pid, return_dict):
        # print(os.path.join(self.tmp_dir, self.dataset_name, str(self.fix_query_times), arm))
        # print(self.is_sharing)
        # logging.disable(sys.maxsize)
        automl = autosklearn.classification.AutoSklearnClassifier(
            include_estimators=[arm],
            include_preprocessors=["no_preprocessing", ],
            exclude_preprocessors=None,
            ensemble_size=0,
            initial_configurations_via_metalearning=0,
            memory_limit=ml_memory_limit,
            n_jobs=None, per_run_time_limit=None,
            time_left_for_this_task=time_limit_per_run, seed=seed,
            tmp_folder=os.path.join(tmp_dir, dataset_name, str(fix_query_times), arm),
            delete_tmp_folder_after_terminate=True,
            output_folder=os.path.join(output_dir, dataset_name, str(fix_query_times), arm),
            delete_output_folder_after_terminate=True,
            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_args,
            metric=metric
        )
        automl.fit(X_train, y_train)
        # logging.disable(logging.NOTSET)
        best_try_idx = np.argmax(automl.cv_results_['mean_test_score'])
        performance = automl.cv_results_['mean_test_score'][best_try_idx]
        best_config = automl.cv_results_['params'][best_try_idx]
        # with open(os.path.join(self.tmp_dir, f'tmp_model_{pid}.pkl'), 'wb') as f:
        #     pickle.dump(automl, f)
        # with open(os.path.join(self.tmp_dir, f'tmp_model_{pid}.pkl'), 'rb') as f:
        #     automl_renew = pickle.load(f)
        model_instance = copy.deepcopy(automl.get_models_with_weights()[0][1])
        return_values = dict()
        return_values['performance'] = performance
        return_values['best_model_config'] = best_config
        return_values['best_model_instance'] = model_instance
        del automl
        return_dict[arm] = return_values
        # return arm, return_values

    @staticmethod
    def _play_arm_parallel_wrapper_with_return(ml_memory_limit, time_limit_per_run, tmp_dir, dataset_name,
                                               fix_query_times, output_dir, resampling_strategy, resampling_args,
                                               is_sharing, metric, seed, X_train, y_train, arm, pid):
        # print(os.path.join(self.tmp_dir, self.dataset_name, str(self.fix_query_times), arm))
        # print(self.is_sharing)
        # logging.disable(sys.maxsize)
        automl = autosklearn.classification.AutoSklearnClassifier(
            include_estimators=[arm],
            include_preprocessors=["no_preprocessing", ],
            exclude_preprocessors=None,
            ensemble_size=0,
            initial_configurations_via_metalearning=0,
            memory_limit=ml_memory_limit,
            n_jobs=None, per_run_time_limit=None,
            time_left_for_this_task=time_limit_per_run, seed=seed,
            tmp_folder=os.path.join(tmp_dir, dataset_name, str(fix_query_times), arm),
            delete_tmp_folder_after_terminate=True,
            output_folder=os.path.join(output_dir, dataset_name, str(fix_query_times), arm),
            delete_output_folder_after_terminate=True,
            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_args,
            metric=metric
        )
        automl.fit(X_train, y_train)
        # logging.disable(logging.NOTSET)
        best_try_idx = np.argmax(automl.cv_results_['mean_test_score'])
        performance = automl.cv_results_['mean_test_score'][best_try_idx]
        best_config = automl.cv_results_['params'][best_try_idx]
        # with open(os.path.join(self.tmp_dir, f'tmp_model_{pid}.pkl'), 'wb') as f:
        #     pickle.dump(automl, f)
        # with open(os.path.join(self.tmp_dir, f'tmp_model_{pid}.pkl'), 'rb') as f:
        #     automl_renew = pickle.load(f)
        model_instance = copy.deepcopy(automl.get_models_with_weights()[0][1])
        return_values = dict()
        return_values['performance'] = performance
        return_values['best_model_config'] = best_config
        return_values['best_model_instance'] = model_instance
        del automl
        # return_dict[arm] = return_values
        return arm, return_values

    def get_stats(self):
        return self.time_records, self.final_rewards

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def score(self, best_model, X_test, y_test):
        if self.metric is None:
            self.logger.info('Metric is set to accuracy_score by default!')
            self.metric = accuracy
        y_pred = best_model.predict(X_test)
        return self.metric(y_test, y_pred)

    def rising_bandit(self, X_train, y_train, lab_id, enalbe_weighting=True):
        """perform rising bandit.

        :param X_train:
        :param y_train:
        """
        if enalbe_weighting:
            # domain D
            cash_valid_list, valid_weights_list = self._train_domain_D(queried_id=lab_id, num_cv=1)
            cash_valid = np.asarray(cash_valid_list[0])
            valid_weights = np.asarray(valid_weights_list[0])
            sorted_val_id_args = np.argsort(cash_valid)
            # s_val_id = cash_valid[sorted_val_id_args]
            s_val_w = valid_weights[sorted_val_id_args]
            s_predefined_id = np.zeros(len(lab_id), dtype=int) - 1
            s_predefined_id[[lab_id.index.index(item) for item in cash_valid]] = 0
        else:
            cash_valid_list, valid_weights_list = self._train_domain_D(queried_id=lab_id, num_cv=1)
            cash_valid = np.asarray(cash_valid_list[0])
            s_predefined_id = np.zeros(len(lab_id), dtype=int) - 1
            s_predefined_id[[lab_id.index.index(item) for item in cash_valid]] = 0
            s_val_w = np.ones(len(cash_valid))

        last_reward = 0
        evolve = False
        for _arm_id in range(2):
            # self.logger.info('Optimize %s in the %d-th iteration' % (_arm, _iter_id))
            # print('Optimize %s in the %d-th iteration' % (_arm, _iter_id))
            if self.current_arm_id + _arm_id >= len(self.arms):
                break
            _arm = self.arms[self.current_arm_id + _arm_id]
            try:
                reward, best_config, model_instance = self.play_arm(X_train.copy(), y_train.copy(),
                                                                    _arm, val_id=s_predefined_id,
                                                                    val_weights=s_val_w)
            except:
                reward = 0
                best_config = None
                model_instance = None
                if _arm == 'gradient_boosting':
                    tp_model = GradientBoostingClassifier()
                    tp_model.fit(X_train.copy(), y_train.copy())
                    model_instance = tp_model
                else:
                    self.current_arm_id += 1
                    _arm = self.arms[self.current_arm_id + _arm_id]
                    reward, best_config, model_instance = self.play_arm(X_train.copy(), y_train.copy(),
                                                                        _arm, val_id=s_predefined_id,
                                                                        val_weights=s_val_w)


            if _arm_id == 1 and reward > last_reward:
                evolve = True
            last_reward = reward
            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            if reward > self.best_perf:
                self.best_perf = reward
                if model_instance is not None:
                    self.best_model = model_instance
                self.best_hpo_config = best_config
            if reward > self.sub_best_reward[_arm]:
                self.sub_best_reward[_arm] = reward
                self.sub_best_model[_arm] = model_instance
            self.logger.info('The best performance found for %s is %.4f' % (_arm, reward))
            # print('The best performance found for %s is %.4f' % (_arm, reward))

        if evolve:
            self.current_arm_id += 1


    def search_and_train_test(self, X_train, y_train, X_test, y_test, lab_id, enalbe_weighting=False):
        self.rising_bandit(X_train, y_train, lab_id=lab_id, enalbe_weighting=enalbe_weighting)
        self.refit(X_train, y_train)
        test_perf = self.score(self.best_model, X_test, y_test)
        self.query_times += 1
        self.is_sharing = True
        return test_perf

    def refit(self, X_train, y_train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctxt_params = signature(self.best_model.fit)
            if 'y' in ctxt_params.parameters:
                self.best_model.fit(X=X_train, y=y_train)
            else:
                self.best_model.fit(X=X_train, Y=y_train)

    @staticmethod
    def _pred_fun(arm_name, model, model_reward, X_train, y_train, X_unlab):
        if model is None or model_reward == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X=X_train, y=y_train)
        pred = model.predict(X_unlab)
        return arm_name, pred

    def _parallel_refit_predict(self, X_train, y_train, X_unlab):
        ret_dict = dict()
        # with multiprocessing.Manager() as manager:
        #     return_dict = manager.dict()
        #     processes = []
        #     logging.disable(sys.maxsize)
        #     for i, arm in enumerate(self.arm_candidate):  # set this at roughly half of your cores
        #         p = multiprocessing.Process(
        #             target=self._pred_fun,
        #             args=(arm, self.sub_best_model[arm],
        #                   self.rewards[arm][-1], X_train.copy(),
        #                   y_train.copy(), X_unlab.copy(),
        #                   return_dict),
        #         )
        #         p.start()
        #         processes.append(p)
        #     for p in processes:
        #         p.join()
        #     logging.disable(logging.NOTSET)
        #     ret_dict.update(return_dict)

        # Pool based parallel
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2)) as pool:
            params_iter = [(arm, self.sub_best_model[arm], self.rewards[arm][-1], X_train, y_train, X_unlab) for i, arm
                           in enumerate(self.arm_candidate)]
            mp_results = pool.starmap(self._pred_fun, params_iter)
            # pool.close()
            # pool.join()
        for rs in mp_results:
            ret_dict[rs[0]] = rs[1]

        return ret_dict

    def query_by_disagreement(self, X, y, lab_ind, unlab_ind, batch_size=1, parallel=True):
        """Accept categorical predictions, not proba

        :param X: array, feature matrix of the WHOLE dataset
        :param y: array, label vector of the WHOLE dataset
        :param lab_ind: array, labeled data indexes
        :param unlab_ind: array, unlabeled data indexes
        :param batch_size: int, batch size
        :return: queried_ind: int, selected data index for querying
        """
        unlab_data = X[unlab_ind]
        # committee_pred = dict()
        committee_size = len(self.arm_candidate)
        if committee_size == 1:  # only one candidate
            return self._query_by_unc(X, y, lab_ind, unlab_ind, batch_size=1)

        lab_size = len(np.unique(y))
        votes = np.zeros((lab_size, len(unlab_ind)))

        if parallel:
            pred_results = self._parallel_refit_predict(X[lab_ind], y[lab_ind], X[unlab_ind])
            committee_size = len(pred_results)
            for arm, pred_ in pred_results.items():
                for i in range(lab_size):
                    votes[i][pred_ == i] += (1 * self.sub_best_reward[arm])
        else:
            for arm in self.arm_candidate:
                # committee_pred[arm] = self.sub_best_model[arm].predict(unlab_data)
                # print("training: ", arm)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.rewards[arm][-1] != 0 and self.sub_best_model[arm] is not None:
                        self.sub_best_model[arm].fit(X=X[lab_ind], y=y[lab_ind])  # refit
                        pred_ = np.asarray(self.sub_best_model[arm].predict(unlab_data))
                        for i in range(lab_size):
                            votes[i][pred_ == i] += (1 * self.sub_best_reward[arm])
                    else:
                        committee_size -= 1

        avoid_zero = votes.copy()
        avoid_zero[avoid_zero <= 0] = 1e-4
        score = [-np.sum(votes[:, i] / committee_size * np.log(avoid_zero[:, i] / committee_size)) for i in
                 range(len(unlab_ind))]

        return [unlab_ind[ind] for ind in nlargestarg(score, batch_size)]

    def query_by_EE(self, X, y, lab_ind, unlab_ind, batch_size=1, trade_off=1):
        unlab_data = X[unlab_ind]
        # committee_pred = dict()
        committee_size = len(self.arm_candidate)
        if committee_size == 1:  # only one candidate
            return self._query_by_unc(X, y, lab_ind, unlab_ind, batch_size=1)

        # QBC
        lab_size = len(np.unique(y))
        votes = np.zeros((lab_size, len(unlab_ind)))
        for arm in self.arm_candidate:
            # committee_pred[arm] = self.sub_best_model[arm].predict(unlab_data)
            # print("training: ", arm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.rewards[arm][-1] != 0 and self.sub_best_model[arm] is not None:
                    ctxt_params = signature(self.sub_best_model[arm].fit)
                    if 'y' in ctxt_params.parameters:
                        self.sub_best_model[arm].fit(X=X[lab_ind], y=y[lab_ind])
                    else:
                        self.sub_best_model[arm].fit(X=X[lab_ind], Y=y[lab_ind])
                    pred_ = np.asarray(self.sub_best_model[arm].predict(unlab_data))
                    for i in range(lab_size):
                        votes[i][pred_ == i] += (1 * self.sub_best_reward[arm])
                else:
                    committee_size -= 1

        qbc_score = [-np.sum(votes[:, i] / committee_size * np.log(votes[:, i] / committee_size + 1e-9)) for i in
                     range(len(unlab_ind))]

        # unc
        self.refit(X_train=X[lab_ind], y_train=y[lab_ind])
        pred = self.best_model.predict_proba(X[unlab_ind])
        unc_score = alipy.query_strategy.QueryInstanceUncertainty(X=X, y=y).calc_entropy(predict_proba=pred)

        return [unlab_ind[ind] for ind in nlargestarg(unc_score + np.asarray(qbc_score) * trade_off, batch_size)]


    def _query_by_unc(self, X, y, lab_ind, unlab_ind, batch_size=1):
        self.refit(X_train=X[lab_ind], y_train=y[lab_ind])
        pred = self.best_model.predict_proba(X[unlab_ind])
        return alipy.query_strategy.QueryInstanceUncertainty(X=X, y=y).select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                                                predict=pred,
                                                                                                batch_size=batch_size)
