from subprocess import check_call
import os
import multiprocessing.dummy


def call_script(*args, **kwargs):
    args = list(args)
    res = check_call(["python", "/path/to/the/source_root/main.py"] + args)
    print(res)


if __name__ == "__main__":
    with multiprocessing.dummy.Pool(processes=os.cpu_count()) as pool:
        datasets = [54, 50, 1501, 36, 40670, 3, 40701, 1489, 28, 4534, 1046, 6]
        folds = list(range(10))
        strategies = ['DUAL', 'random_cash_successive', 'inas', 'ALMS']
        tradeoff = -1.0
        save_home = "/path/to/save_home"

        params_iter = []
        for da in datasets:
            for strat in strategies:
                for fd in folds:
                    pa = ("--dataset", f"{da}", "--start_fold", f"{fd}",
                          "--end_fold", f"{fd + 1}", "--save_home", f"{save_home}",
                          "--tradeoff", f"{tradeoff}", "--strategy", f"{strat}")
                    params_iter.append(pa)

        mp_results = pool.starmap(call_script, params_iter)
