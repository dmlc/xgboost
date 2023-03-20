Python API Reference
====================
This page gives the Python API reference of xgboost, please also refer to Python Package Introduction for more information about the Python package.

.. contents::
  :backlinks: none
  :local:

Global Configuration
--------------------
.. autofunction:: xgboost.config_context

.. autofunction:: xgboost.set_config

.. autofunction:: xgboost.get_config

Core Data Structure
-------------------
.. automodule:: xgboost.core

.. autoclass:: xgboost.DMatrix
    :members:
    :show-inheritance:

.. autoclass:: xgboost.QuantileDMatrix
    :show-inheritance:

.. autoclass:: xgboost.Booster
    :members:
    :show-inheritance:


Learning API
------------
.. automodule:: xgboost.training

.. autofunction:: xgboost.train

.. autofunction:: xgboost.cv


Scikit-Learn API
----------------

.. automodule:: xgboost.sklearn
.. autoclass:: xgboost.XGBRegressor
    :members:
    :inherited-members:
    :show-inheritance:
.. autoclass:: xgboost.XGBClassifier
    :members:
    :inherited-members:
    :show-inheritance:
.. autoclass:: xgboost.XGBRanker
    :members:
    :inherited-members:
    :show-inheritance:
.. autoclass:: xgboost.XGBRFRegressor
    :members:
    :inherited-members:
    :show-inheritance:
.. autoclass:: xgboost.XGBRFClassifier
    :members:
    :inherited-members:
    :show-inheritance:

Plotting API
------------
.. automodule:: xgboost.plotting

.. autofunction:: xgboost.plot_importance

.. autofunction:: xgboost.plot_tree

.. autofunction:: xgboost.to_graphviz

.. _callback_api:

Callback API
------------
.. automodule:: xgboost.callback
.. autoclass:: xgboost.callback.TrainingCallback
    :members:

.. autoclass:: xgboost.callback.EvaluationMonitor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.callback.EarlyStopping
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.callback.LearningRateScheduler
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.callback.TrainingCheckPoint
    :members:
    :inherited-members:
    :show-inheritance:

.. _dask_api:

Dask API
--------
.. automodule:: xgboost.dask

.. autoclass:: xgboost.dask.DaskDMatrix
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.dask.DaskQuantileDMatrix
    :members:
    :inherited-members:
    :show-inheritance:

.. autofunction:: xgboost.dask.train

.. autofunction:: xgboost.dask.predict

.. autofunction:: xgboost.dask.inplace_predict

.. autoclass:: xgboost.dask.DaskXGBClassifier
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.dask.DaskXGBRegressor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.dask.DaskXGBRanker
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.dask.DaskXGBRFRegressor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.dask.DaskXGBRFClassifier
    :members:
    :inherited-members:
    :show-inheritance:


PySpark API
-----------

.. automodule:: xgboost.spark

.. autoclass:: xgboost.spark.SparkXGBClassifier
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.spark.SparkXGBClassifierModel
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.spark.SparkXGBRegressor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.spark.SparkXGBRegressorModel
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.spark.SparkXGBRanker
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: xgboost.spark.SparkXGBRankerModel
    :members:
    :inherited-members:
    :show-inheritance:
