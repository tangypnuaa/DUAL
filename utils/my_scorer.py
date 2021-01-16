from sklearn.metrics import accuracy_score


class weighted_acc:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, y, y_pred, **kwargs):
        if len(y_pred) == len(self.weights):
            return accuracy_score(y_true=y, y_pred=y_pred, sample_weight=self.weights)
        else:
            return accuracy_score(y_true=y, y_pred=y_pred)
