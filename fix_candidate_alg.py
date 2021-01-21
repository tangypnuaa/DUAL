from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


def get_vote_result(lab_arr, weights):
    # print(lab_arr)
    # print(weights)
    vote = dict()
    for ires, res in enumerate(lab_arr):
        if hasattr(res, '__iter__'):
            res = res[0]
        if res in vote.keys():
            vote[res] += weights[ires]
        else:
            vote[res] = weights[ires]
        # print("ok2")
    sort_key = sorted(vote.items(), key=lambda kv:kv[1], reverse=True)
    # print(sort_key)
    return sort_key[0][0]


class ALMS_alg:
    def __init__(self, X, y, initial_labid):
        self.classifiers = [
            SGDClassifier(),
            PassiveAggressiveClassifier(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            SVC(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            ExtraTreeClassifier(),
            MLPClassifier(),
            AdaBoostClassifier(n_estimators=20),
            RandomForestClassifier(n_estimators=30),
            # GradientBoostingClassifier()
        ]
        self.trained_classifiers = [None] * len(self.classifiers)
        self.X = X
        self.y = y
        self.validation_set = list(initial_labid)
        self.model_weights = [1] * len(self.classifiers)
        # if len(self.validation_set) > 100:
        #     self.validation_set = list(np.random.choice(self.validation_set, size=100, replace=False))
        self.training_set = []

    def _calc_training(self, unlab_x, unlab_y):
        s = np.asarray([1.0 / len(self.validation_set)] * len(self.validation_set))
        P = np.zeros((len(self.validation_set), len(self.classifiers)))
        L = np.zeros((len(self.classifiers), len(self.validation_set)))

        for i in range(len(self.validation_set)):
            Pm, Lm = self._get_tr_fast(i, unlab_x, unlab_y)
            P[i, :] = Pm
            L[:, i] = Lm
        return s @ P @ L @ s.T

    def _get_tr_fast(self, i, unlab_x, unlab_y):
        """
        distribution of model j on example i.

        :param i:
        :param j:
        :return:
        """
        loss_arr = []
        tr_id = copy.copy(self.validation_set)
        tr_id.pop(i)
        for model in self.classifiers:
            train_feat = np.vstack((self.X[self.training_set], self.X[tr_id], unlab_x))
            train_y = np.hstack((self.y[self.training_set], self.y[tr_id], unlab_y))
            try:
                model.fit(X=train_feat, y=train_y)
                valx1 = self.X[self.validation_set[i]]
                valy1 = self.y[self.validation_set[i]]
                pred = model.predict(valx1.reshape((1, -1)))
                if pred == valy1:  # use 0/1 loss
                    loss = 0
                else:
                    loss = 1
            except:
                loss = 1
            loss_arr.append(loss)
        L = loss_arr
        exp_loss = np.exp(loss_arr)
        M = 1 - exp_loss / (np.sum(exp_loss) + 1e-9)
        return M, L

    def _calc_selection(self, unlab_x, unlab_y):
        s = np.zeros(len(self.validation_set) + 1) + 1 / (len(self.validation_set) + 1)
        P = np.zeros((len(self.validation_set) + 1, len(self.classifiers)))
        L = np.zeros((len(self.classifiers), len(self.validation_set) + 1))
        for i in range(len(self.validation_set) + 1):
            Pm, Lm = self._get_se_fast(i, unlab_x, unlab_y)
            P[i, :] = Pm
            L[:, i] = Lm
        return s @ P @ L @ s.T

    def _get_se_fast(self, i, unlab_x, unlab_y):
        """
        distribution of model j on example i.

        :param i:
        :param j:
        :return:
        """
        loss_arr = []
        tr_id = copy.copy(self.validation_set)
        if i != len(tr_id):
            tr_id.pop(i)
        for model in self.classifiers:
            train_feat = np.vstack((self.X[self.training_set], self.X[tr_id]))
            train_y = np.hstack((self.y[self.training_set], self.y[tr_id]))
            try:
                model.fit(X=train_feat, y=train_y)
                if i != len(self.validation_set):
                    valx1 = self.X[self.validation_set[i]]
                    valy1 = self.y[self.validation_set[i]]
                else:
                    valx1 = unlab_x
                    valy1 = unlab_y
                pred = model.predict(valx1.reshape((1, -1)))
                if pred == valy1:  # use 0/1 loss
                    loss = 0
                else:
                    loss = 1
            except:
                loss = 1
            loss_arr.append(loss)
        L = loss_arr
        exp_loss = np.exp(loss_arr)
        M = 1 - exp_loss / (np.sum(exp_loss) + 1e-9)
        return M, L

    def select(self, unlab_ind):
        tr_arr = []
        se_arr = []
        for un_id in unlab_ind:
            unlab_x = self.X[un_id]
            lab_arr = []
            for im, model in enumerate(self.trained_classifiers):
                if self.model_weights[im] != 0 and model is not None:
                    try:
                        plab = model.predict(unlab_x.reshape((1, -1)))
                    except:
                        plab = 0
                        self.model_weights[im] = 0
                else:
                    plab = 0
                lab_arr.append(plab)
            unlab_y = [get_vote_result(lab_arr, self.model_weights)]

            tr = self._calc_training(unlab_x, unlab_y)
            se = self._calc_selection(unlab_x, unlab_y)
            tr_arr.append(tr)
            se_arr.append(se)
        if max(tr_arr) > max(se_arr):
            qid = unlab_ind[np.argmax(tr_arr)]
            self.training_set.append(qid)
        else:
            qid = np.random.choice(unlab_ind, size=1, replace=False)[0]
            self.validation_set.append(qid)
        return qid

    def train_test(self, test_idx):
        model_val_perf_arr = []
        for im, model in enumerate(self.classifiers):
            train_feat = self.X[self.validation_set]
            train_y = self.y[self.validation_set]
            try:
                results = cross_validate(estimator=model, X=train_feat, y=train_y, cv=2, scoring=["accuracy"],
                                         return_train_score=False, n_jobs=1)
                mean_acc = np.mean(results['test_accuracy'])
                model.fit(train_feat, train_y)
                self.trained_classifiers[im] = model
            except:
                mean_acc = 0
            self.model_weights[im] = mean_acc

            model_val_perf_arr.append(mean_acc)

        best_model_id = np.argmax(model_val_perf_arr)
        test_model = self.classifiers[best_model_id]
        test_model.fit(X=self.X[self.validation_set + self.training_set],
                       y=self.y[self.validation_set + self.training_set])
        pred = test_model.predict(self.X[test_idx])
        return accuracy_score(pred, self.y[test_idx])

