## Dual Active Learning for Both Model and Data Selection

Published at IJCAI-21

Authors: Ying-Peng Tang and Sheng-Jun Huang

This work proposes a novel framework of DUal Active Learning (DUAL) to simultaneously perform model search and data selection.

The model search method is implemented based on `auto-sklearn` package, and the `alipy` package is employed to implement the compared data querying strategies.

## Main requirements

* **Linux-like operating system** (required by auto-sklearn)
* **python >= 3.6**
* **auto-sklearn == 0.12.0** (Please follow the [official instruction](https://automl.github.io/auto-sklearn/master/installation.html) for installation)
* **alipy**

## Other implementation details

* Hyperparameter spaces are set as the default values of auto-sklearn. (See `autosklearn/pipeline/components/classification` for the specific settings.)
* The domain discriminator D is implemented by a 3-layer neural network. (See L201 in `algorithm.py`)
* Complexity order of candidate models for Active-iNAS are defined in L130 in `ALMS.py`

## Usage
### Run DUAL, CASH, ALMS, Active-iNAS methods

* From cmd line (check the tunable parameters in main.py)
```
python main.py --dataset 50 --strategy DUAL --save_home /path/to/save
```
```
python main.py --dataset 50 --strategy random_cash_successive --save_home /path/to/save
```

* Run with multiprocessing (please set the parameters inside pshell.py)

```
python pshell.py
```

### Run the other compared methods

You must run DUAL first to save the target model, which is searched on initially labeled data. Then the following cmd could be work. (Please set the parameter `tmp_home` inside.)

```
python compared_methods/baselines.py
```

### Please cite our work

```
Tang, Y.P. and Huang, S.J., 2021. Dual Active Learning for Both Model and Data Selection. In IJCAI (pp. 3052-3058).
```

```
@inproceedings{tang2021dual,
  title={Dual Active Learning for Both Model and Data Selection.},
  author={Tang, Ying-Peng and Huang, Sheng-Jun},
  booktitle={Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence},
  pages={3052--3058},
  year={2021}
}
```
