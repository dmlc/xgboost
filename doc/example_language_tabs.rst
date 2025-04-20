****************************
Example with Language Tabs
****************************

This is an example of a tutorial with language tabs for both Python and R.

.. tabbed:: Python

    This section contains Python code examples.

    .. code-block:: python

        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        
        # Load data
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters
        params = {
            'max_depth': 3,
            'eta': 0.3,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        # Train model
        num_round = 10
        bst = xgb.train(params, dtrain, num_round)
        
        # Make prediction
        preds = bst.predict(dtrain)

.. tabbed:: R

    This section contains R code examples.

    .. code-block:: r

        library(xgboost)
        
        # Load data
        data(agaricus.train, package='xgboost')
        train <- agaricus.train
        
        # Create DMatrix
        dtrain <- xgb.DMatrix(train$data, label=train$label)
        
        # Set parameters
        params <- list(
            max_depth = 3,
            eta = 0.3,
            objective = "binary:logistic",
            eval_metric = "logloss"
        )
        
        # Train model
        num_round = 10
        bst <- xgb.train(params, dtrain, num_round)
        
        # Make prediction
        preds <- predict(bst, dtrain) 