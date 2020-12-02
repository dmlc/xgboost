Python API Reference
====================
This page gives the Python API reference of xgboost, please also refer to Python Package Introduction for more information about python package.

.. contents::
  :backlinks: none
  :local:

Core Data Structure
-------------------
.. automodule:: xgboost.core

.. autoclass:: xgboost.DMatrix
    :members:
    :show-inheritance:

.. autoclass:: xgboost.DeviceQuantileDMatrix
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
.. autofunction:: xgboost.callback.TrainingCallback

.. autofunction:: xgboost.callback.EvaluationMonitor

.. autofunction:: xgboost.callback.EarlyStopping

.. autofunction:: xgboost.callback.LearningRateScheduler

.. autofunction:: xgboost.callback.TrainingCheckPoint

.. _dask_api:

Dask API
--------
.. automodule:: xgboost.dask

.. autofunction:: xgboost.dask.DaskDMatrix

.. autofunction:: xgboost.dask.DaskDeviceQuantileDMatrix

.. autofunction:: xgboost.dask.train

.. autofunction:: xgboost.dask.predict

.. autofunction:: xgboost.dask.inplace_predict

.. autofunction:: xgboost.dask.DaskXGBClassifier

.. autofunction:: xgboost.dask.DaskXGBRegressor
