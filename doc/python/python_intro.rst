###########################
Python Package Introduction
###########################

This document gives a basic walkthrough of the xgboost package for Python.  The Python
package is consisted of 3 different interfaces, including native interface, scikit-learn
interface and dask interface.  For introduction to dask interface please see
:doc:`/tutorials/dask`.

**List of other Helpful Links**

* :doc:`/python/examples/index`
* :doc:`Python API Reference <python_api>`

**Contents**

.. contents::
  :backlinks: none
  :local:

Install XGBoost
---------------
To install XGBoost, follow instructions in :doc:`/install`.

To verify your installation, run the following in Python:

.. code-block:: python

  import xgboost as xgb

.. _python_data_interface:

Data Interface
--------------
The XGBoost python module is able to load data from many different types of data format,
including:

- NumPy 2D array
- SciPy 2D sparse array
- Pandas data frame
- cuDF DataFrame
- cupy 2D array
- dlpack
- datatable
- XGBoost binary buffer file.
- LIBSVM text format file
- Comma-separated values (CSV) file

(See :doc:`/tutorials/input_format` for detailed description of text input format.)

The data is stored in a :py:class:`DMatrix <xgboost.DMatrix>` object.

* To load a NumPy array into :py:class:`DMatrix <xgboost.DMatrix>`:

  .. code-block:: python

    data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    label = np.random.randint(2, size=5)  # binary target
    dtrain = xgb.DMatrix(data, label=label)

* To load a :py:mod:`scipy.sparse` array into :py:class:`DMatrix <xgboost.DMatrix>`:

  .. code-block:: python

    csr = scipy.sparse.csr_matrix((dat, (row, col)))
    dtrain = xgb.DMatrix(csr)

* To load a Pandas data frame into :py:class:`DMatrix <xgboost.DMatrix>`:

  .. code-block:: python

    data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
    label = pandas.DataFrame(np.random.randint(2, size=4))
    dtrain = xgb.DMatrix(data, label=label)

* Saving :py:class:`DMatrix <xgboost.DMatrix>` into a XGBoost binary file will make loading faster:

  .. code-block:: python

    dtrain = xgb.DMatrix('train.svm.txt')
    dtrain.save_binary('train.buffer')

* Missing values can be replaced by a default value in the :py:class:`DMatrix <xgboost.DMatrix>` constructor:

  .. code-block:: python

    dtrain = xgb.DMatrix(data, label=label, missing=np.NaN)

* Weights can be set when needed:

  .. code-block:: python

    w = np.random.rand(5, 1)
    dtrain = xgb.DMatrix(data, label=label, missing=np.NaN, weight=w)

When performing ranking tasks, the number of weights should be equal
to number of groups.

* To load a LIBSVM text file or a XGBoost binary file into :py:class:`DMatrix <xgboost.DMatrix>`:

  .. code-block:: python

    dtrain = xgb.DMatrix('train.svm.txt')
    dtest = xgb.DMatrix('test.svm.buffer')

  The parser in XGBoost has limited functionality. When using Python interface, it's
  recommended to use sklearn ``load_svmlight_file`` or other similar utilites than
  XGBoost's builtin parser.

* To load a CSV file into :py:class:`DMatrix <xgboost.DMatrix>`:

  .. code-block:: python

    # label_column specifies the index of the column containing the true label
    dtrain = xgb.DMatrix('train.csv?format=csv&label_column=0')
    dtest = xgb.DMatrix('test.csv?format=csv&label_column=0')

  The parser in XGBoost has limited functionality. When using Python interface, it's
  recommended to use pandas ``read_csv`` or other similar utilites than XGBoost's builtin
  parser.


Setting Parameters
------------------
XGBoost can use either a list of pairs or a dictionary to set :doc:`parameters </parameter>`. For instance:

* Booster parameters

  .. code-block:: python

    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'

* You can also specify multiple eval metrics:

  .. code-block:: python

    param['eval_metric'] = ['auc', 'ams@0']

    # alternatively:
    # plst = param.items()
    # plst += [('eval_metric', 'ams@0')]

* Specify validations set to watch performance

  .. code-block:: python

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

Training
--------

Training a model requires a parameter list and data set.

.. code-block:: python

  num_round = 10
  bst = xgb.train(param, dtrain, num_round, evallist)

After training, the model can be saved.

.. code-block:: python

  bst.save_model('0001.model')

The model and its feature map can also be dumped to a text file.

.. code-block:: python

  # dump model
  bst.dump_model('dump.raw.txt')
  # dump model with feature map
  bst.dump_model('dump.raw.txt', 'featmap.txt')

A saved model can be loaded as follows:

.. code-block:: python

  bst = xgb.Booster({'nthread': 4})  # init model
  bst.load_model('model.bin')  # load data

Methods including `update` and `boost` from `xgboost.Booster` are designed for
internal usage only.  The wrapper function `xgboost.train` does some
pre-configuration including setting up caches and some other parameters.

Early Stopping
--------------
If you have a validation set, you can use early stopping to find the optimal number of boosting rounds.
Early stopping requires at least one set in ``evals``. If there's more than one, it will use the last.

.. code-block:: python

  train(..., evals=evals, early_stopping_rounds=10)

The model will train until the validation score stops improving. Validation error needs to decrease at least every ``early_stopping_rounds`` to continue training.

If early stopping occurs, the model will have two additional fields: ``bst.best_score``, ``bst.best_iteration``.  Note that :py:meth:`xgboost.train` will return a model from the last iteration, not the best one.

This works with both metrics to minimize (RMSE, log loss, etc.) and to maximize (MAP, NDCG, AUC). Note that if you specify more than one evaluation metric the last one in ``param['eval_metric']`` is used for early stopping.

Prediction
----------
A model that has been trained or loaded can perform predictions on data sets.

.. code-block:: python

  # 7 entities, each contains 10 features
  data = np.random.rand(7, 10)
  dtest = xgb.DMatrix(data)
  ypred = bst.predict(dtest)

If early stopping is enabled during training, you can get predictions from the best iteration with ``bst.best_iteration``:

.. code-block:: python

  ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

Plotting
--------

You can use plotting module to plot importance and output tree.

To plot importance, use :py:meth:`xgboost.plot_importance`. This function requires ``matplotlib`` to be installed.

.. code-block:: python

  xgb.plot_importance(bst)

To plot the output tree via ``matplotlib``, use :py:meth:`xgboost.plot_tree`, specifying the ordinal number of the target tree. This function requires ``graphviz`` and ``matplotlib``.

.. code-block:: python

  xgb.plot_tree(bst, num_trees=2)

When you use ``IPython``, you can use the :py:meth:`xgboost.to_graphviz` function, which converts the target tree to a ``graphviz`` instance. The ``graphviz`` instance is automatically rendered in ``IPython``.

.. code-block:: python

  xgb.to_graphviz(bst, num_trees=2)


Scikit-Learn interface
----------------------

XGBoost provides an easy to use scikit-learn interface for some pre-defined models
including regression, classification and ranking.

.. code-block:: python

  # Use "gpu_hist" for training the model.
  reg = xgb.XGBRegressor(tree_method="gpu_hist")
  # Fit the model using predictor X and response y.
  reg.fit(X, y)
  # Save model into JSON format.
  reg.save_model("regressor.json")

User can still access the underlying booster model when needed:

.. code-block:: python

   booster: xgb.Booster = reg.get_booster()
