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

Plotting API
------------
.. automodule:: xgboost.plotting

.. autofunction:: xgboost.plot_importance

.. autofunction:: xgboost.plot_tree

.. autofunction:: xgboost.to_graphviz

.. _callback_api:

Callback API
------------
.. autofunction:: xgboost.callback.print_evaluation

.. autofunction:: xgboost.callback.record_evaluation

.. autofunction:: xgboost.callback.reset_learning_rate

.. autofunction:: xgboost.callback.early_stop
