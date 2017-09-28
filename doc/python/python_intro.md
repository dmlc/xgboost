Python Package Introduction
===========================
This document gives a basic walkthrough of xgboost python package.

***List of other Helpful Links***
* [Python walkthrough code collections](https://github.com/tqchen/xgboost/blob/master/demo/guide-python)
* [Python API Reference](python_api.rst)

Install XGBoost
---------------
To install XGBoost, do the following:

* Run `make` in the root directory of the project
* In the  `python-package` directory, run
```shell
python setup.py install
```

To verify your installation, try to `import xgboost` in Python.
```python
import xgboost as xgb
```

Data Interface
--------------
The XGBoost python module is able to load data from:
- libsvm txt format file
- Numpy 2D array, and
- xgboost binary buffer file.

The data is stored in a ```DMatrix``` object.

* To load a libsvm text file or a XGBoost binary file into ```DMatrix```:
```python
dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.buffer')
```
* To load a numpy array into ```DMatrix```:
```python
data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)
```
* To load a scpiy.sparse array into ```DMatrix```:
```python
csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)
```
* Saving ```DMatrix``` into a XGBoost binary file will make loading faster:
```python
dtrain = xgb.DMatrix('train.svm.txt')
dtrain.save_binary('train.buffer')
```
* Missing values can be replaced by a default value in the ```DMatrix``` constructor:
```python
dtrain = xgb.DMatrix(data, label=label, missing=-999.0)
```
* Weights can be set when needed:
```python
w = np.random.rand(5, 1)
dtrain = xgb.DMatrix(data, label=label, missing=-999.0, weight=w)
```

Setting Parameters
------------------
XGBoost can use either a list of pairs or a dictionary to set [parameters](../parameter.md). For instance:
* Booster parameters
```python
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
```
* You can also specify multiple eval metrics:
```python
param['eval_metric'] = ['auc', 'ams@0']

# alternatively:
# plst = param.items()
# plst += [('eval_metric', 'ams@0')]
```

* Specify validations set to watch performance
```python
evallist = [(dtest, 'eval'), (dtrain, 'train')]
```

Training
--------

Training a model requires a parameter list and data set.
```python
num_round = 10
bst = xgb.train(plst, dtrain, num_round, evallist)
```
After training, the model can be saved.
```python
bst.save_model('0001.model')
```
The model and its feature map can also be dumped to a text file.
```python
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt', 'featmap.txt')
```
A saved model can be loaded as follows:
```python
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('model.bin')  # load data
```

Early Stopping
--------------
If you have a validation set, you can use early stopping to find the optimal number of boosting rounds.
Early stopping requires at least one set in `evals`. If there's more than one, it will use the last.

`train(..., evals=evals, early_stopping_rounds=10)`

The model will train until the validation score stops improving. Validation error needs to decrease at least every `early_stopping_rounds` to continue training.

If early stopping occurs, the model will have three additional fields: `bst.best_score`, `bst.best_iteration` and `bst.best_ntree_limit`. Note that `train()` will return a model from the last iteration, not the best one.

This works with both metrics to minimize (RMSE, log loss, etc.) and to maximize (MAP, NDCG, AUC). Note that if you specify more than one evaluation metric the last one in `param['eval_metric']` is used for early stopping.

Prediction
----------
A model that has been trained or loaded can perform predictions on data sets.
```python
# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)
```

If early stopping is enabled during training, you can get predictions from the best iteration with `bst.best_ntree_limit`:
```python
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
```

Plotting
--------

You can use plotting module to plot importance and output tree.

To plot importance, use ``plot_importance``. This function requires ``matplotlib`` to be installed.

```python
xgb.plot_importance(bst)
```

To plot the output tree via ``matplotlib``, use ``plot_tree``, specifying the ordinal number of the target tree. This function requires ``graphviz`` and ``matplotlib``.

```python
xgb.plot_tree(bst, num_trees=2)
```

When you use ``IPython``, you can use the ``to_graphviz`` function, which converts the target tree to a ``graphviz`` instance. The ``graphviz`` instance is automatically rendered in ``IPython``.

```python
xgb.to_graphviz(bst, num_trees=2)
```
