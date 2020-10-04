##################
Callback Functions
##################

This document gives a basic walkthrough of callback function used in XGBoost Python
package.  In XGBoost 1.3, a new callback interface is designed for Python package, which
provides the flexiablity of designing various extension for training.  Also, XGBoost has a
number of pre-defined callbacks for supporting early stopping, checkpoints etc.

#######################
Using builtin callbacks
#######################

By default, training methods in XGBoost have parameters like ``early_stopping_rounds`` and
``verbose``/``verbose_eval``, when specified the training procedure will define the
corresponding callbacks internally.  For example, when ``early_stopping_rounds`` is
specified, ``EarlyStopping`` callback is invoked inside iteration loop.  You can also pass
this callback function directly into XGBoost:

.. code-block:: python

    D_train = xgb.DMatrix(X_train, y_train)
    D_valid = xgb.DMatrix(X_valid, y_valid)

    # Define a custom evaluation metric used for early stopping.
    def eval_error_metric(predt, dtrain: xgb.DMatrix):
        label = dtrain.get_label()
        r = np.zeros(predt.shape)
        gt = predt > 0.5
        r[gt] = 1 - label[gt]
        le = predt <= 0.5
        r[le] = label[le]
        return 'PyError', np.sum(r)

    # Specify which dataset and which metric should be used for early stopping.
    early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                            metric=tm.eval_error_metric,
                                            metric_name='PyError',
                                            data_name='Valid')
    booster = xgb.train(
        {'objective': 'binary:logistic',
         'eval_metric': ['error', 'rmse'],
         'tree_method': 'hist'}, D_train,
        evals=[(D_train, 'Train'), (D_valid, 'Valid')],
        num_boost_round=1000,
        callbacks=[early_stop],
        verbose_eval=False)

    dump = booster.get_dump(dump_format='json')
    assert len(early_stop.stopping_history['Valid']['PyError']) == len(dump)

##########################
Defining your own callback
##########################

In here we will define a callback for monitoring shap value changes during training.
First XGBoost provides an interface class: ``xgboost.callback.TrainingCallback``, user
defined callbacks should inherit this class and override corresponding methods.

.. code-block:: python
    pass


The full example is in.
