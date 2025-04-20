#############################
Introduction to Boosted Trees
#############################
XGBoost stands for "Extreme Gradient Boosting", where the term "Gradient Boosting" originates from the paper *Greedy Function Approximation: A Gradient Boosting Machine*, by Friedman.

The term **gradient boosted trees** has been around for a while, and there are a lot of materials on the topic.
This tutorial will explain boosted trees in a self-contained and principled way using the elements of supervised learning.
We think this explanation is cleaner, more formal, and motivates the model formulation used in XGBoost.

*******************************
Elements of Supervised Learning
*******************************
XGBoost is used for supervised learning problems, where we use the training data (with multiple features) :math:`x_i` to predict a target variable :math:`y_i`.
Before we learn about trees specifically, let us start by reviewing the basic elements in supervised learning.

Model and Parameters
====================
The **model** in supervised learning usually refers to the mathematical structure of by which the prediction :math:`y_i` is made from the input :math:`x_i`.
A common example is a *linear model*, where the prediction is given as :math:`\hat{y}_i = \sum_j \theta_j x_{ij}`, a linear combination of weighted input features.
The prediction value can have different interpretations, depending on the task, i.e., regression or classification.
For example, it can be logistic transformed to get the probability of positive class in logistic regression, and it can also be used as a ranking score when we want to rank the outputs.

The **parameters** are the undetermined part that we need to learn from data. In linear regression problems, the parameters are the coefficients :math:`\theta`.
Usually we will use :math:`\theta` to denote the parameters (there are many parameters in a model, our definition here is sloppy).

Objective Function: Training Loss + Regularization
==================================================
With judicious choices for :math:`y_i`, we may express a variety of tasks, such as regression, classification, and ranking.
The task of **training** the model amounts to finding the best parameters :math:`\theta` that best fit the training data :math:`x_i` and labels :math:`y_i`. In order to train the model, we need to define the **objective function**
to measure how well the model fit the training data.

A salient characteristic of objective functions is that they consist of two parts: **training loss** and **regularization term**:

.. math::

  \text{obj}(\theta) = L(\theta) + \Omega(\theta)

where :math:`L` is the training loss function, and :math:`\Omega` is the regularization term. The training loss measures how *predictive* our model is with respect to the training data.
A common choice of :math:`L` is the *mean squared error*, which is given by

.. math::

  L(\theta) = \sum_i (y_i-\hat{y}_i)^2

Another commonly used loss function is logistic loss, to be used for logistic regression:

.. math::

  L(\theta) = \sum_i[ y_i\ln (1+e^{-\hat{y}_i}) + (1-y_i)\ln (1+e^{\hat{y}_i})]

The **regularization term** is what people usually forget to add. The regularization term controls the complexity of the model, which helps us to avoid overfitting.
This sounds a bit abstract, so let us consider the following problem in the following picture. You are asked to *fit* visually a step function given the input data points
on the upper left corner of the image.
Which solution among the three do you think is the best fit?

.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/step_fit.png
  :alt: step functions to fit data points, illustrating bias-variance tradeoff

The correct answer is marked in red. Please consider if this visually seems a reasonable fit to you. The general principle is we want both a *simple* and *predictive* model.
The tradeoff between the two is also referred as **bias-variance tradeoff** in machine learning.

Why introduce the general principle?
====================================
The elements introduced above form the basic elements of supervised learning, and they are natural building blocks of machine learning toolkits.
For example, you should be able to describe the differences and commonalities between gradient boosted trees and random forests.
Understanding the process in a formalized way also helps us to understand the objective that we are learning and the reason behind the heuristics such as
pruning and smoothing.

***********************
Decision Tree Ensembles
***********************
Now that we have introduced the elements of supervised learning, let us get started with real trees.
To begin with, let us first learn about the model choice of XGBoost: **decision tree ensembles**.
The tree ensemble model consists of a set of classification and regression trees (CART). Here's a simple example of a CART that classifies whether someone will like a hypothetical computer game X.

.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/cart.png
  :width: 100%
  :alt: a toy example for CART

We classify the members of a family into different leaves, and assign them the score on the corresponding leaf.
A CART is a bit different from decision trees, in which the leaf only contains decision values. In CART, a real score
is associated with each of the leaves, which gives us richer interpretations that go beyond classification.
This also allows for a principled, unified approach to optimization, as we will see in a later part of this tutorial.

Usually, a single tree is not strong enough to be used in practice. What is actually used is the ensemble model,
which sums the prediction of multiple trees together.

.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/twocart.png
  :width: 100%
  :alt: a toy example for tree ensemble, consisting of two CARTs

Here is an example of a tree ensemble of two trees. The prediction scores of each individual tree are summed up to get the final score.
If you look at the example, an important fact is that the two trees try to *complement* each other.
Mathematically, we can write our model in the form

.. math::

  \hat{y}_i = \sum_{k=1}^K f_k(x_i), f_k \in \mathcal{F}

where :math:`K` is the number of trees, :math:`f_k` is a function in the functional space :math:`\mathcal{F}`, and :math:`\mathcal{F}` is the set of all possible CARTs. The objective function to be optimized is given by

.. math::

  \text{obj}(\theta) = \sum_i^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \omega(f_k)

where :math:`\omega(f_k)` is the complexity of the tree :math:`f_k`, defined in detail later.

Now here comes a trick question: what is the *model* used in random forests? Tree ensembles! So random forests and boosted trees are really the same models; the
difference arises from how we train them. This means that, if you write a predictive service for tree ensembles, you only need to write one and it should work
for both random forests and gradient boosted trees. (See `Treelite <https://treelite.readthedocs.io/en/latest/index.html>`_ for an actual example.) One example of why elements of supervised learning rock.

*************
Tree Boosting
*************
Now that we introduced the model, let us turn to training: How should we learn the trees?
The answer is, as is always for all supervised learning models: *define an objective function and optimize it*!

Let the following be the objective function (remember it always needs to contain training loss and regularization):

.. math::

  \text{obj} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^t\omega(f_k)

in which :math:`t` is the number of trees in our ensemble.
(Each training step will add one new tree, so that at step :math:`t` the ensemble contains :math:`K=t` trees).

Additive Training
=================

The first question we want to ask: what are the **parameters** of trees?
You can find that what we need to learn are those functions :math:`f_k`, each containing the structure
of the tree and the leaf scores. Learning tree structure is much harder than traditional optimization problem where you can simply take the gradient.
It is intractable to learn all the trees at once.
Instead, we use an additive strategy: fix what we have learned, and add one new tree at a time.
We write the prediction value at step :math:`t` as :math:`\hat{y}_i^{(t)}`. Then we have

.. math::

  \hat{y}_i^{(0)} &= 0\\
  \hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
  \hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
  &\dots\\
  \hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)

It remains to ask: which tree do we want at each step?  A natural thing is to add the one that optimizes our objective.

Training a Boosted Tree with XGBoost
====================================

.. tabbed:: Python

    .. code-block:: python

        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import load_boston
        from sklearn.metrics import mean_squared_error
        
        # Load data
        boston = load_boston()
        X = boston.data
        y = boston.target
        
        # Split data into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters
        params = {
            'max_depth': 3,  # maximum depth of a tree
            'eta': 0.3,      # learning rate
            'objective': 'reg:squarederror',  # regression task
            'eval_metric': 'rmse'  # evaluation metric
        }
        
        # Train model
        num_round = 100  # number of trees
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)
        
        # Make prediction
        preds = bst.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"RMSE: {rmse:.4f}")
        
        # Feature importance
        importance = bst.get_score(importance_type='weight')
        print("Feature importance (weight):")
        for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"Feature {key}: {value}")

.. tabbed:: R

    .. code-block:: r

        library(xgboost)
        library(Matrix)
        library(caret)
        
        # Load Boston Housing dataset
        data(Boston, package = "MASS")
        
        # Create train/test split
        set.seed(123)
        train_idx <- createDataPartition(Boston$medv, p = 0.8, list = FALSE)
        train_data <- Boston[train_idx, ]
        test_data <- Boston[-train_idx, ]
        
        # Prepare data matrices
        train_x <- data.matrix(train_data[, -14])  # all but the target column
        train_y <- train_data$medv
        test_x <- data.matrix(test_data[, -14])
        test_y <- test_data$medv
        
        # Create DMatrix
        dtrain <- xgb.DMatrix(data = train_x, label = train_y)
        dtest <- xgb.DMatrix(data = test_x, label = test_y)
        
        # Set parameters
        params <- list(
            max_depth = 3,          # maximum depth of a tree
            eta = 0.3,              # learning rate
            objective = "reg:squarederror",  # regression task
            eval_metric = "rmse"    # evaluation metric
        )
        
        # Train model
        watchlist <- list(train = dtrain, eval = dtest)
        bst <- xgb.train(
            params = params,
            data = dtrain,
            nrounds = 100,
            watchlist = watchlist,
            early_stopping_rounds = 10,
            verbose = 1
        )
        
        # Make prediction
        preds <- predict(bst, dtest)
        rmse <- sqrt(mean((test_y - preds)^2))
        cat("RMSE:", round(rmse, 4), "\n")
        
        # Feature importance
        importance <- xgb.importance(feature_names = colnames(train_x), model = bst)
        print("Feature importance (weight):")
        head(importance, 5) 