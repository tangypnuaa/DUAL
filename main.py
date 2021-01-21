import argparse
import os
import sys
import time
import random
from active_cash import active_learning, random_cash, random_cash_successive
from ALMS import ALMS_AL, active_inas

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='50', help="openml dataset id")
    parser.add_argument('--metric', type=str, default='accuracy')
    # AL
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--end_fold', type=int, default=10)
    parser.add_argument('--al_folds', type=int, default=10)
    parser.add_argument('--test_ratio', type=float, default=0.4)
    parser.add_argument('--ini_lab_ratio', type=float, default=0.05)
    parser.add_argument('--query_budget', type=int, default=300)
    parser.add_argument('--tradeoff', type=float, default=-1.0,
                        help="The value of beta, -1.0 means 1/g as presented in the paper.")
    parser.add_argument('--strategy', type=str, default="DUAL", help="['DUAL', 'random', 'random_cash_successive']."
                                                                     "random means R. w/0 suc. in ablation studies, "
                                                                     "random_cash_successive means R. CASH.")
    # bandit
    parser.add_argument('--arms_level', type=int, default=3, help="How many arms will be used in automl. "
                                                                  "default is 3 (use all arms).")
    parser.add_argument('--bandit_trial_budget', type=int, default=1)
    parser.add_argument('--smbo_time_limit_per_run', type=int, default=60, help="time budget for each trial of SMAC")
    parser.add_argument('--ml_memory_limit', type=int, default=10240, help="Memory limit (MB) for each CASH instance.")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--alpha', type=int, default=10,
                        help="The latest x reward will be used to calculate the upper and lower bounds. "
                             "It is the C parameter in the paper.")
    # saving
    parser.add_argument('--save_home', type=str, default="./output/")
    # pipeline
    parser.add_argument('--enable_removing_iter', type=int, default=10,
                        help="After how many queries to start removing arms")

    args = parser.parse_args()
    args.resampling_strategy = 'cv'
    args.resampling_args = 3
    args.process_limit = False

    print(args)
    _saving_home = args.save_home
    data_save_dir = _saving_home + "openml"
    df_al_save_dir = _saving_home + 'al_out'
    df_cash_save_dir = _saving_home + 'autosklearn_out'
    df_cash_tmp_dir = _saving_home + 'autosklearn_tmp'

    print(f'==> Start to perform active learning for {args.dataset}.')
    time.sleep(random.random() * 5)
    if args.strategy == "DUAL":
        resutls = active_learning(dataset_id=args.dataset, arms_level=args.arms_level, al_folds=args.al_folds,
                                  test_ratio=args.test_ratio, ini_lab_ratio=args.ini_lab_ratio, data_home=data_save_dir,
                                  bandit_trial_budget=args.bandit_trial_budget, ml_memory_limit=args.ml_memory_limit,
                                  smbo_time_limit_per_run=args.smbo_time_limit_per_run, parallel=args.parallel,
                                  query_budget=args.query_budget, al_save_dir=df_al_save_dir,
                                  cash_save_dir=df_cash_save_dir, cash_tmp_dir=df_cash_tmp_dir,
                                  resampling_strategy=args.resampling_strategy, resampling_args=args.resampling_args,
                                  enable_removing_iter=args.enable_removing_iter,
                                  alpha=args.alpha, start_fold=args.start_fold, end_fold=args.end_fold,
                                  process_limit=args.process_limit, tradeoff=args.tradeoff)
    elif args.strategy == "random":
        resutls = random_cash(dataset_id=args.dataset, arms_level=args.arms_level, al_folds=args.al_folds,
                              test_ratio=args.test_ratio, ini_lab_ratio=args.ini_lab_ratio, data_home=data_save_dir,
                              bandit_trial_budget=args.bandit_trial_budget, ml_memory_limit=args.ml_memory_limit,
                              smbo_time_limit_per_run=args.smbo_time_limit_per_run, parallel=args.parallel,
                              query_budget=args.query_budget, al_save_dir=df_al_save_dir,
                              cash_save_dir=df_cash_save_dir, cash_tmp_dir=df_cash_tmp_dir,
                              resampling_strategy=args.resampling_strategy, resampling_args=args.resampling_args,
                              enable_removing_iter=args.enable_removing_iter,
                              alpha=args.alpha, start_fold=args.start_fold, end_fold=args.end_fold,
                              process_limit=args.process_limit, tradeoff=args.tradeoff)
    elif args.strategy == "random_cash_successive":
        resutls = random_cash_successive(dataset_id=args.dataset, arms_level=args.arms_level,
                                         al_folds=args.al_folds,
                                         test_ratio=args.test_ratio, ini_lab_ratio=args.ini_lab_ratio,
                                         data_home=data_save_dir,
                                         bandit_trial_budget=args.bandit_trial_budget,
                                         ml_memory_limit=args.ml_memory_limit,
                                         smbo_time_limit_per_run=args.smbo_time_limit_per_run,
                                         parallel=args.parallel,
                                         query_budget=args.query_budget, al_save_dir=df_al_save_dir,
                                         cash_save_dir=df_cash_save_dir, cash_tmp_dir=df_cash_tmp_dir,
                                         resampling_strategy=args.resampling_strategy,
                                         resampling_args=args.resampling_args,
                                         enable_removing_iter=args.enable_removing_iter,
                                         alpha=args.alpha, start_fold=args.start_fold, end_fold=args.end_fold,
                                         process_limit=args.process_limit, tradeoff=args.tradeoff)
    elif args.strategy == "ALMS":
        results = ALMS_AL(dataset_id=args.dataset, arms_level=args.arms_level,
                          al_folds=args.al_folds,
                          test_ratio=args.test_ratio, ini_lab_ratio=args.ini_lab_ratio,
                          data_home=data_save_dir,
                          bandit_trial_budget=args.bandit_trial_budget,
                          ml_memory_limit=args.ml_memory_limit,
                          smbo_time_limit_per_run=args.smbo_time_limit_per_run,
                          parallel=args.parallel,
                          query_budget=args.query_budget, al_save_dir=df_al_save_dir,
                          cash_save_dir=df_cash_save_dir, cash_tmp_dir=df_cash_tmp_dir,
                          resampling_strategy=args.resampling_strategy,
                          resampling_args=args.resampling_args,
                          enable_removing_iter=args.enable_removing_iter,
                          alpha=args.alpha, start_fold=args.start_fold, end_fold=args.end_fold,
                          process_limit=args.process_limit, tradeoff=args.tradeoff)
    elif args.strategy == "inas":
        results = active_inas(dataset_id=args.dataset, arms_level=args.arms_level,
                              al_folds=args.al_folds,
                              test_ratio=args.test_ratio, ini_lab_ratio=args.ini_lab_ratio,
                              data_home=data_save_dir,
                              bandit_trial_budget=args.bandit_trial_budget,
                              ml_memory_limit=args.ml_memory_limit,
                              smbo_time_limit_per_run=args.smbo_time_limit_per_run,
                              parallel=args.parallel,
                              query_budget=args.query_budget, al_save_dir=df_al_save_dir,
                              cash_save_dir=df_cash_save_dir, cash_tmp_dir=df_cash_tmp_dir,
                              resampling_strategy=args.resampling_strategy,
                              resampling_args=args.resampling_args,
                              enable_removing_iter=args.enable_removing_iter,
                              alpha=args.alpha, start_fold=args.start_fold, end_fold=args.end_fold,
                              process_limit=args.process_limit, tradeoff=args.tradeoff)
