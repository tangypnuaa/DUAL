import os
import re
import pickle

def get_last_model(al_home, dataset_id, dataset_name, fold):
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
        'sgd': 'SGD'
    }

    saver_path = os.path.join(al_home, str(dataset_id) + '_' + dataset_name, f'AL_round_{fold}.pkl')
    with open(saver_path, 'rb') as f:
        saver = pickle.load(f)
    last_state = saver[len(saver)-1]
    conf = last_state['best_config']

    prefix = 'classifier:'
    args_dict = dict()
    model_name = conf['classifier:__choice__']
    for k in conf.keys():
        if k.startswith(f"{prefix}{model_name}:"):
            arg_name = re.sub(f"{prefix}{model_name}:*", "", k)
            if conf[k] == 'False':
                args_dict[arg_name] = False
            elif conf[k] == 'True':
                args_dict[arg_name] = True
            else:
                args_dict[arg_name] = conf[k]

    exec(f"from autosklearn.pipeline.components.classification.{model_name} import {name2name[model_name]}")
    model = eval(f"{name2name[model_name]}(**args_dict)")
    return model


if __name__ == "__main__":
    tmp_home = "./output/"
    al_home = tmp_home + "al_out"
    data_home = tmp_home + "openml"
    data_id = 54
    data_name = 'vehicle'
    method = "QueryInstanceRandom"
    folds = 4
    get_last_model(al_home=al_home, dataset_id=data_id, dataset_name=data_name, fold=0)
