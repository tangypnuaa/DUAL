import os
import numpy as np
import pickle
import alipy
import warnings
import multiprocessing
from active_cash import fetch_dataset
from sklearn.metrics import accuracy_score
from alipy.index.index_collections import IndexCollection
from compared_methods.best_models import get_last_model
from inspect import signature

class ComparedAL:
    def __init__(self, dataset_id, al_home, data_home, folds, model_type=0):
        self.dataset_id = str(dataset_id)
        self.al_home = al_home
        self.data_home = data_home
        self.folds = folds
        self.model_type = model_type    # 0 ini model, 1 last model
        # read data, datasplit, initial model
        self.dataset_name, self.X, self.y = fetch_dataset(self.dataset_id, data_home=data_home)
        if 10000 > len(self.X) > 3000:
            self.query_budget = 1000
        elif len(self.X) >= 10000:
            self.query_budget = 2000
        else:
            self.query_budget = 300
        self.alibox = alipy.ToolBox.load(os.path.join(al_home,
                                                      dataset_id + '_' + self.dataset_name,
                                                      'al_settings.pkl'))
        try:
            self.train_idxs, self.test_idxs, self.label_inds, self.unlab_inds = self.alibox.get_split()
        except:
            self.train_idxs, self.test_idxs, self.label_inds, self.unlab_inds = alipy.data_manipulate.split_load(
                os.path.join(al_home, self.dataset_id + '_' + self.dataset_name))
        if len(np.shape(self.train_idxs)) == 1:
            self.train_idxs = self.train_idxs.reshape((1, -1))
            self.test_idxs = self.test_idxs.reshape((1, -1))
            self.label_inds = self.label_inds.reshape((1, -1))
            self.unlab_inds = self.unlab_inds.reshape((1, -1))

    def run_fix_model(self, strategy_name, budget=180, i=0, verbose=True):
        # print("start to perform active learning: ", self.strategy_name)
        if self.model_type == 0:
            with open(os.path.join(al_home, self.dataset_id + '_' + self.dataset_name,
                                   f'im_{self.dataset_name}_{i}_t-1.0.pkl'), 'rb') as f:
                model = pickle.load(f)
        else:
            model = get_last_model(al_home=al_home, dataset_id=self.dataset_id,
                                   dataset_name=self.dataset_name, fold=i)
        train_idx, test_idx, label_ind, unlab_ind = self.train_idxs[i], self.test_idxs[i], IndexCollection(
            self.label_inds[i]), IndexCollection(self.unlab_inds[i])
        if strategy_name in {"QueryInstanceQUIRE", "QueryInstanceCoresetGreedy"}:
            strategy = self.alibox.get_query_strategy(strategy_name=strategy_name, train_idx=train_idx)
        else:
            strategy = self.alibox.get_query_strategy(strategy_name=strategy_name)
        saver = alipy.experiment.StateIO(i, train_idx, test_idx, label_ind, unlab_ind, verbose=verbose,
                                         saving_path=os.path.join(al_home,
                                                                  self.dataset_id + '_' + self.dataset_name,
                                                                  strategy_name + '_f' + str(i) +
                                                                  f'{1 if self.model_type else ""}.pkl'))

        # init point
        ctxt_params = signature(model.fit)
        if 'y' in ctxt_params.parameters:
            model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
        else:
            model.fit(X=self.X[label_ind.index, :], Y=self.y[label_ind.index])
        pred = model.predict(self.X[test_idx, :])
        accuracy = accuracy_score(y_true=self.y[test_idx], y_pred=pred)
        saver.set_initial_point(accuracy)

        for ib in range(budget):
            # Select a subset of Uind according to the query strategy
            select_ind = strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=1, model=model)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            if 'y' in ctxt_params.parameters:
                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
            else:
                model.fit(X=self.X[label_ind.index, :], Y=self.y[label_ind.index])
            pred = model.predict(self.X[test_idx, :])
            accuracy = accuracy_score(y_true=self.y[test_idx], y_pred=pred)

            # Save intermediate results to file
            st = alipy.experiment.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()

    def run_parallel(self, strategy_name, budget, folds):
        processes = []
        tar_func = self.run_fix_model
        for i in range(folds):
            p = multiprocessing.Process(
                target=tar_func,
                args=(strategy_name, budget, i, False),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


tmp_home = "/path/to/save_home/"
al_home = tmp_home + "al_out"
data_home = tmp_home + "openml"
for id in [54, 50, 1501, 36, 40670, 3, 40701, 1489, 28, 4534, 1046, 6]:
    print("dataset:", id)
    cal = ComparedAL(dataset_id=str(id), al_home=al_home, data_home=data_home, folds=10)
    for strat in ["QueryInstanceCoresetGreedy", "QueryInstanceUncertainty", "QueryInstanceQUIRE", "QueryInstanceRandom"]:
        print("method:", strat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.run_parallel(strategy_name=strat, budget=cal.query_budget, folds=10)
