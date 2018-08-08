#########################
Notes on Parameter Tuning
#########################
Parameter tuning is a dark art in machine learning, the optimal parameters
of a model can depend on many scenarios. So it is impossible to create a
comprehensive guide for doing so.

This document tries to provide some guideline for parameters in XGBoost.

************************************
Understanding Bias-Variance Tradeoff
************************************
If you take a machine learning or statistics course, this is likely to be one
of the most important concepts.
When we allow the model to get more complicated (e.g. more depth), the model
has better ability to fit the training data, resulting in a less biased model.
However, such complicated model requires more data to fit.

Most of parameters in XGBoost are about bias variance tradeoff. The best model
should trade the model complexity with its predictive power carefully.
:doc:`Parameters Documentation </parameter>` will tell you whether each parameter
will make the model more conservative or not. This can be used to help you
turn the knob between complicated model and simple model.

*******************
Control Overfitting
*******************
When you observe high training accuracy, but low test accuracy, it is likely that you encountered overfitting problem.

There are in general two ways that you can control overfitting in XGBoost:

* The first way is to directly control model complexity.

  - This includes ``max_depth``, ``min_child_weight`` and ``gamma``.

* The second way is to add randomness to make training robust to noise.

  - This includes ``subsample`` and ``colsample_bytree``.
  - You can also reduce stepsize ``eta``. Remember to increase ``num_round`` when you do so.

*************************
Handle Imbalanced Dataset
*************************
For common cases such as ads clickthrough log, the dataset is extremely imbalanced.
This can affect the training of XGBoost model, and there are two ways to improve it.

* If you care only about the overall performance metric (AUC) of your prediction

  - Balance the positive and negative weights via ``scale_pos_weight``
  - Use AUC for evaluation

* If you care about predicting the right probability

  - In such a case, you cannot re-balance the dataset
  - Set parameter ``max_delta_step`` to a finite number (say 1) to help convergence
