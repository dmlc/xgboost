#####
Model
#####

Slice tree model
----------------

When ``booster`` is set to ``gbtree`` or ``dart``, XGBoost builds a tree model, which is a
list of trees and can be sliced into multiple sub-models.

.. code-block:: python
    from sklearn.datasets import make_classification
    num_classes = 3
    X, y = make_classification(n_samples=1000, n_informative=5,
                               n_classes=num_classes)
    dtrain = xgb.DMatrix(data=X, label=y)
    num_parallel_tree = 4
    num_boost_round = 16
    total_trees = num_parallel_tree * num_classes * num_boost_round

    # We build a boosted random forest for classification here.
    booster = xgb.train({
        'num_parallel_tree': 4, 'subsample': 0.5, 'num_class': 3},
                        num_boost_round=num_boost_round, dtrain=dtrain)

    # This is the sliced model, containing [3, 7) forests
    sliced: xgb.Booster = booster[3:7]
