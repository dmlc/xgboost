###############################################
Survival Analysis with Accelerated Failure Time
###############################################

.. contents::
  :local:
  :backlinks: none

**************************
What is survival analysis?
**************************

**Survival analysis (regression)** models **time to an event of interest**. Survival analysis is a special kind of regression and differs from the conventional regression task as follows:

* The label is always positive, since you cannot wait a negative amount of time until the event occurs.
* The label may not be fully known, or **censored**, because "it takes time to measure time."

The second bullet point is crucial and we should dwell on it more. As you may have guessed from the name, one of the earliest applications of survival analysis is to model mortality of a given population. Let's take `NCCTG Lung Cancer Dataset <https://stat.ethz.ch/R-manual/R-devel/library/survival/html/lung.html>`_ as an example. The first 8 columns represent features and the last column, Time to death, represents the label.

==== === === ======= ======== ========= ======== ======= ========================
Inst Age Sex ph.ecog ph.karno pat.karno meal.cal wt.loss **Time to death (days)**
==== === === ======= ======== ========= ======== ======= ========================
3    74  1   1       90       100       1175     N/A     306
3    68  1   0       90       90        1225     15      455
3    56  1   0       90       90        N/A      15      :math:`[1010, +\infty)`
5    57  1   1       90       60        1150     11      210
1    60  1   0       100      90        N/A      0       883
12   74  1   1       50       80        513      0       :math:`[1022, +\infty)`
7    68  2   2       70       60        384      10      310
==== === === ======= ======== ========= ======== ======= ========================

Take a close look at the label for the third patient. **His label is a range, not a single number.** The third patient's label is said to be **censored**, because for some reason the experimenters could not get a complete measurement for that label. One possible scenario: the patient survived the first 1010 days and walked out of the clinic on the 1011th day, so his death was not directly observed. Another possibility: The experiment was cut short (since you cannot run it forever) before his death could be observed. In any case, his label is :math:`[1010, +\infty)`, meaning his time to death can be any number that's higher than 1010, e.g. 2000, 3000, or 10000.

There are four kinds of censoring:

* **Uncensored**: the label is not censored and given as a single number.
* **Right-censored**: the label is of form :math:`[a, +\infty)`, where :math:`a` is the lower bound.
* **Left-censored**: the label is of form :math:`[0, b]`, where :math:`b` is the upper bound.
* **Interval-censored**: the label is of form :math:`[a, b]`, where :math:`a` and :math:`b` are the lower and upper bounds, respectively.

Right-censoring is the most commonly used.

******************************
Accelerated Failure Time model
******************************
**Accelerated Failure Time (AFT)** model is one of the most commonly used models in survival analysis. The model is of the following form:

.. math::

  \ln{Y} = \langle \mathbf{w}, \mathbf{x} \rangle + \sigma Z

where

* :math:`\mathbf{x}` is a vector in :math:`\mathbb{R}^d` representing the features.
* :math:`\mathbf{w}` is a vector consisting of :math:`d` coefficients, each corresponding to a feature.
* :math:`\langle \cdot, \cdot \rangle` is the usual dot product in :math:`\mathbb{R}^d`.
* :math:`\ln{(\cdot)}` is the natural logarithm.
* :math:`Y` and :math:`Z` are random variables.

  - :math:`Y` is the output label.
  - :math:`Z` is a random variable of a known probability distribution. Common choices are the normal distribution, the logistic distribution, and the extreme distribution. Intuitively, :math:`Z` represents the "noise" that pulls the prediction :math:`\langle \mathbf{w}, \mathbf{x} \rangle` away from the true log label :math:`\ln{Y}`.

* :math:`\sigma` is a parameter that scales the size of :math:`Z`.

Note that this model is a generalized form of a linear regression model :math:`Y = \langle \mathbf{w}, \mathbf{x} \rangle`. In order to make AFT work with gradient boosting, we revise the model as follows:

.. math::

  \ln{Y} = \mathcal{T}(\mathbf{x}) + \sigma Z

where :math:`\mathcal{T}(\mathbf{x})` represents the output from a decision tree ensemble, given input :math:`\mathbf{x}`. Since :math:`Z` is a random variable, we have a likelihood defined for the expression :math:`\ln{Y} = \mathcal{T}(\mathbf{x}) + \sigma Z`. So the goal for XGBoost is to maximize the (log) likelihood by fitting a good tree ensemble :math:`\mathcal{T}(\mathbf{x})`.

**********
How to use
**********
The first step is to express the labels in the form of a range, so that **every data point has two numbers associated with it, namely the lower and upper bounds for the label.** For uncensored labels, use a degenerate interval of form :math:`[a, a]`.

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

================= ==================== =================== ===================
Censoring type    Interval form        Lower bound finite? Upper bound finite?
================= ==================== =================== ===================
Uncensored        :math:`[a, a]`       |tick|              |tick|
Right-censored    :math:`[a, +\infty)` |tick|              |cross|
Left-censored     :math:`[0, b]`       |tick|              |tick|
Interval-censored :math:`[a, b]`       |tick|              |tick|
================= ==================== =================== ===================

Collect the lower bound numbers in one array (let's call it ``y_lower_bound``) and the upper bound number in another array (call it ``y_upper_bound``). The ranged labels are associated with a data matrix object via calls to :meth:`xgboost.DMatrix.set_float_info`:

.. code-block:: python
  :caption: Python

  import numpy as np
  import xgboost as xgb

  # 4-by-2 Data matrix
  X = np.array([[1, -1], [-1, 1], [0, 1], [1, 0]])
  dtrain = xgb.DMatrix(X)
  
  # Associate ranged labels with the data matrix.
  # This example shows each kind of censored labels.
  #                         uncensored    right     left  interval
  y_lower_bound = np.array([      2.0,     3.0,     0.0,     4.0])
  y_upper_bound = np.array([      2.0, +np.inf,     4.0,     5.0])
  dtrain.set_float_info('label_lower_bound', y_lower_bound)
  dtrain.set_float_info('label_upper_bound', y_upper_bound)

.. code-block:: r
  :caption: R
  
  library(xgboost)

  # 4-by-2 Data matrix
  X <- matrix(c(1., -1., -1., 1., 0., 1., 1., 0.),
              nrow=4, ncol=2, byrow=TRUE)
  dtrain <- xgb.DMatrix(X)

  # Associate ranged labels with the data matrix.
  # This example shows each kind of censored labels.
  #                   uncensored  right  left  interval
  y_lower_bound <- c(        2.,    3.,   0.,       4.)
  y_upper_bound <- c(        2.,  +Inf,   4.,       5.)
  setinfo(dtrain, 'label_lower_bound', y_lower_bound)
  setinfo(dtrain, 'label_upper_bound', y_upper_bound)

Now we are ready to invoke the training API:

.. code-block:: python
  :caption: Python

  params = {'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 1.20,
            'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 2}
  bst = xgb.train(params, dtrain, num_boost_round=5,
                  evals=[(dtrain, 'train')])

.. code-block:: r
  :caption: R

  params <- list(objective='survival:aft',
                 eval_metric='aft-nloglik',
                 aft_loss_distribution='normal',
                 aft_loss_distribution_scale=1.20,
                 tree_method='hist',
                 learning_rate=0.05,
                 max_depth=2)
  watchlist <- list(train = dtrain)
  bst <- xgb.train(params, dtrain, nrounds=5, watchlist)

We set ``objective`` parameter to ``survival:aft`` and ``eval_metric`` to ``aft-nloglik``, so that the log likelihood for the AFT model would be maximized. (XGBoost will actually minimize the negative log likelihood, hence the name ``aft-nloglik``.)

The parameter ``aft_loss_distribution`` corresponds to the distribution of the :math:`Z` term in the AFT model, and ``aft_loss_distribution_scale`` corresponds to the scaling factor :math:`\sigma`.

Currently, you can choose from three probability distributions for ``aft_loss_distribution``:

========================= ===========================================
``aft_loss_distribution`` Probabilty Density Function (PDF)
========================= ===========================================
``normal``                :math:`\dfrac{\exp{(-z^2/2)}}{\sqrt{2\pi}}`
``logistic``              :math:`\dfrac{e^z}{(1+e^z)^2}`
``extreme``               :math:`e^z e^{-\exp{z}}`
========================= ===========================================

Note that it is not yet possible to set the ranged label using the scikit-learn interface (e.g. :class:`xgboost.XGBRegressor`). For now, you should use :class:`xgboost.train` with :class:`xgboost.DMatrix`.
