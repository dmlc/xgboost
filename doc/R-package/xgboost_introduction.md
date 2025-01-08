# XGBoost for R introduction


# Introduction

**XGBoost** is an optimized distributed gradient boosting library
designed to be highly **efficient**, **flexible** and **portable**. It
implements machine learning algorithms under the [Gradient
Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that
solve many data science problems in a fast and accurate way. The same
code runs on major distributed environment (Hadoop, SGE, MPI) and can
solve problems beyond billions of examples.

For an introduction to the concept of gradient boosting, see the
tutorial [Introduction to Boosted
Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) in
XGBoost’s online docs.

For more details about XGBoost’s features and usage, see the [online
documentation](https://xgboost.readthedocs.io/en/stable/) which contains
more tutorials, examples, and details.

This short vignette outlines the basic usage of the R interface for
XGBoost, assuming the reader has some familiarity with the underlying
concepts behind statistical modeling with gradient-boosted decision
trees.

# Building a predictive model

At its core, XGBoost consists of a C++ library which offers bindings for
different programming languages, including R. The R package for XGBoost
provides an idiomatic interface similar to those of other statistical
modeling packages using and x/y design, as well as a lower-level
interface that interacts more directly with the underlying core library
and which is similar to those of other language bindings like Python,
plus various helpers to interact with its model objects such as by
plotting their feature importances or converting them to other formats.

The main function of interest is `xgboost(x, y, ...)`, which calls the
XGBoost model building procedure on observed data of
covariates/features/predictors “x”, and a response variable “y” - it
should feel familiar to users of packages like `glmnet` or `ncvreg`:

``` r
library(xgboost)
data(ToothGrowth)

y <- ToothGrowth$supp # the response which we want to model/predict
x <- ToothGrowth[, c("len", "dose")] # the features from which we want to predct it
model <- xgboost(x, y, nthreads = 1, nrounds = 2)
model
```

    XGBoost model object
    Call:
      xgboost(x = x, y = y, nrounds = 2, nthreads = 1)
    Objective: binary:logistic
    Number of iterations: 2
    Number of features: 2
    Classes: OJ, VC
    XGBoost model object
    Call:
      xgboost(x = x, y = y, nrounds = 2, nthreads = 1)
    Objective: binary:logistic
    Number of iterations: 2
    Number of features: 2
    Classes: OJ, VC

In this case, the “y” response variable that was supplied is a “factor”
type with two classes (“OJ” and “VC”) - hence, XGBoost builds a binary
classification model for it based on the features “x”, by finding a
maximum likelihood estimate (similar to the `faimily="binomial"` model
from R’s `glm` function) through rule buckets obtained from the sum of
two decision trees (from `nrounds=2`), from which we can then predict
probabilities, log-odds, class with highest likelihood, among others:

``` r
predict(model, x[1:6, ], type = "response") # probabilities for y's last level ("VC")
predict(model, x[1:6, ], type = "raw")      # log-odds
predict(model, x[1:6, ], type = "class")    # class with highest probability
```

1  
0.6596265435218812

0.5402158498764043

0.6596265435218814

0.6596265435218815

0.6596265435218816

0.495350033044815

<!-- -->

1  
0.6616302728652952

0.1612115055322653

0.6616302728652954

0.6616302728652955

0.6616302728652956

-0.0186003148555756

1.  VC
2.  VC
3.  VC
4.  VC
5.  VC
6.  OJ

**Levels**: 1. ‘OJ’ 2. ‘VC’

Compared to R’s `glm` function which follows the concepts of “families”
and “links” from GLM theory to fit models for different kinds of
response distributions, XGBoost follows the simpler concept of
“objectives” which mix both of them into one, and which just like `glm`,
allow modeling very different kinds of response distributions
(e.g. discrete choices, real-valued numbers, counts, censored
measurements, etc.) through a common framework.

XGBoost will automatically determine a suitable objective for the
response given its object class (can pass factors for classification,
numeric vectors for regression, `Surv` objects from the `survival`
package for survival, etc. - see `?xgboost` for more details), but this
can be controlled manually through an `objective` parameter based the
kind of model that is desired:

``` r
data(mtcars)

y <- mtcars$mpg
x <- mtcars[, -1]
model_gaussian <- xgboost(x, y, nthreads = 1, nrounds = 2) # default is squared loss (Gaussian)
model_poisson <- xgboost(x, y, objective = "count:poisson", nthreads = 1, nrounds = 2)
model_abserr <- xgboost(x, y, objective = "reg:absoluteerror", nthreads = 1, nrounds = 2)
```

*Note: the objective must match with the type of the “y” response
variable - for example, classification objectives for discrete choices
require “factor” types, while regression models for real-valued data
require “numeric” types.*

# Model parameters

XGBoost models allow a large degree of control over how they are built.
By their nature, gradient-boosted decision tree ensembles are able to
capture very complex patterns between features in the data and a
response variable, which also means they can suffer from overfitting if
not controlled appropirately.

For best results, one needs to find suitable parameters for the data
being modeled. Note that XGBoost does not adjust its default
hyperparameters based on the data, and different datasets will require
vastly different hyperparameters for optimal predictive performance.

For example, for a small dataset like “TootGrowth” which has only two
features and 60 observations, the defaults from XGBoost are an overkill
which lead to severe overfitting - for such data, one might want to have
smaller trees (i.e. more convervative decision rules, capturing simpler
patterns) and fewer of them, for example.

Parameters can be controlled by passing additional arguments to
`xgboost()`. See `?xgb.params` for details about what parameters are
available to control.

``` r
y <- ToothGrowth$supp
x <- ToothGrowth[, c("len", "dose")]
model_conservative <- xgboost(
    x, y, nthreads = 1,
    nrounds = 5,
    max_depth = 2,
    reg_lambda = 0.5,
    learning_rate = 0.15
)
pred_conservative <- predict(
    model_conservative,
    x
)
pred_conservative[1:6] # probabilities are all closer to 0.5 now
```

1  
0.6509257555007932

0.4822041690349583

0.6509257555007934

0.6509257555007935

0.6509257555007936

0.447792500257492

XGBoost also allows the possibility of calculating evaluation metrics
for model quality over boosting rounds, with a wide variety of built-in
metrics available to use. It’s possible to automatically set aside a
fraction of the data to use as evaluation set, from which one can then
visually monitor progress and overfitting:

``` r
xgboost(
    x, y, nthreads = 1,
    eval_set = 0.2,
    monitor_training = TRUE,
    verbosity = 1,
    eval_metric = c("auc", "logloss"),
    nrounds = 5,
    max_depth = 2,
    reg_lambda = 0.5,
    learning_rate = 0.15
)
```

    [1] train-auc:0.763021  train-logloss:0.665634  eval-auc:0.444444   eval-logloss:0.697723 
    [2] train-auc:0.802083  train-logloss:0.643556  eval-auc:0.527778   eval-logloss:0.695267 
    [3] train-auc:0.793403  train-logloss:0.625402  eval-auc:0.472222   eval-logloss:0.701788 
    [4] train-auc:0.815972  train-logloss:0.611023  eval-auc:0.527778   eval-logloss:0.703274 
    [5] train-auc:0.815972  train-logloss:0.599548  eval-auc:0.527778   eval-logloss:0.706069 

    XGBoost model object
    Call:
      xgboost(x = x, y = y, nrounds = 5, verbosity = 1, monitor_training = TRUE, 
        eval_set = 0.2, nthreads = 1, eval_metric = c("auc", "logloss"), 
        max_depth = 2, reg_lambda = 0.5, learning_rate = 0.15)
    Objective: binary:logistic
    Number of iterations: 5
    Number of features: 2
    Classes: OJ, VC
    XGBoost model object
    Call:
      xgboost(x = x, y = y, nrounds = 5, verbosity = 1, monitor_training = TRUE, 
        eval_set = 0.2, nthreads = 1, eval_metric = c("auc", "logloss"), 
        max_depth = 2, reg_lambda = 0.5, learning_rate = 0.15)
    Objective: binary:logistic
    Number of iterations: 5
    Number of features: 2
    Classes: OJ, VC

# Examining model objects

XGBoost model objects for the most part consist of a pointer to a C++
object where most of the information is held and which is interfaced
through the utility functions and methods in the package, but also
contains some R attributes that can be retrieved (and new ones added)
through `attributes()`:

``` r
attributes(model)
```

    $call
    xgboost(x = x, y = y, nrounds = 2, nthreads = 1)

    $params
    $params$objective
    [1] "binary:logistic"

    $params$nthread
    [1] 1

    $params$seed
    [1] 0

    $params$verbosity
    [1] 0

    $params$validate_parameters
    [1] TRUE


    $names
    [1] "ptr"

    $class
    [1] "xgboost"     "xgb.Booster"

    $metadata
    $metadata$y_levels
    [1] "OJ" "VC"

    $metadata$n_targets
    [1] 1

In addition to R attributes (which can be arbitrary R objects), it may
also keep some standardized C-level attributes that one can access and
modify (but which can only be JSON-format):

``` r
xgb.attributes(model)
```

(they are empty for this model)

… but usually, when it comes to getting something out of a model object,
one would typically want to do this through the built-in utility
functions. Some examples:

``` r
xgb.importance(model)
```

A data.table: 2 × 4

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Feature &lt;chr&gt;</th>
<th>Gain &lt;dbl&gt;</th>
<th>Cover &lt;dbl&gt;</th>
<th>Frequency &lt;dbl&gt;</th>
</tr>
</thead>
<tbody>
<tr>
<td>len</td>
<td>0.7444265</td>
<td>0.6830449</td>
<td>0.7333333</td>
</tr>
<tr>
<td>dose</td>
<td>0.2555735</td>
<td>0.3169551</td>
<td>0.2666667</td>
</tr>
</tbody>
</table>

``` r
xgb.model.dt.tree(model)
```

A data.table: 32 × 10

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th>Tree &lt;int&gt;</th>
<th>Node &lt;int&gt;</th>
<th>ID &lt;chr&gt;</th>
<th>Feature &lt;chr&gt;</th>
<th>Split &lt;dbl&gt;</th>
<th>Yes &lt;chr&gt;</th>
<th>No &lt;chr&gt;</th>
<th>Missing &lt;chr&gt;</th>
<th>Gain &lt;dbl&gt;</th>
<th>Cover &lt;dbl&gt;</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0</td>
<td>0-0</td>
<td>len</td>
<td>19.7</td>
<td>0-1</td>
<td>0-2</td>
<td>0-2</td>
<td>5.88235283</td>
<td>15.000000</td>
</tr>
<tr>
<td>0</td>
<td>1</td>
<td>0-1</td>
<td>dose</td>
<td>1.0</td>
<td>0-3</td>
<td>0-4</td>
<td>0-4</td>
<td>2.50230217</td>
<td>7.500000</td>
</tr>
<tr>
<td>0</td>
<td>2</td>
<td>0-2</td>
<td>dose</td>
<td>2.0</td>
<td>0-5</td>
<td>0-6</td>
<td>0-6</td>
<td>2.50230217</td>
<td>7.500000</td>
</tr>
<tr>
<td>0</td>
<td>3</td>
<td>0-3</td>
<td>len</td>
<td>8.2</td>
<td>0-7</td>
<td>0-8</td>
<td>0-8</td>
<td>5.02710962</td>
<td>4.750000</td>
</tr>
<tr>
<td>0</td>
<td>4</td>
<td>0-4</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.36000001</td>
<td>2.750000</td>
</tr>
<tr>
<td>0</td>
<td>5</td>
<td>0-5</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.36000001</td>
<td>2.750000</td>
</tr>
<tr>
<td>0</td>
<td>6</td>
<td>0-6</td>
<td>len</td>
<td>29.5</td>
<td>0-9</td>
<td>0-10</td>
<td>0-10</td>
<td>0.93020594</td>
<td>4.750000</td>
</tr>
<tr>
<td>0</td>
<td>7</td>
<td>0-7</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.36000001</td>
<td>1.500000</td>
</tr>
<tr>
<td>0</td>
<td>8</td>
<td>0-8</td>
<td>len</td>
<td>10.0</td>
<td>0-11</td>
<td>0-12</td>
<td>0-12</td>
<td>0.60633492</td>
<td>3.250000</td>
</tr>
<tr>
<td>0</td>
<td>9</td>
<td>0-9</td>
<td>len</td>
<td>24.5</td>
<td>0-13</td>
<td>0-14</td>
<td>0-14</td>
<td>0.78028417</td>
<td>3.750000</td>
</tr>
<tr>
<td>0</td>
<td>10</td>
<td>0-10</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.15000001</td>
<td>1.000000</td>
</tr>
<tr>
<td>0</td>
<td>11</td>
<td>0-11</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.30000001</td>
<td>1.000000</td>
</tr>
<tr>
<td>0</td>
<td>12</td>
<td>0-12</td>
<td>len</td>
<td>13.6</td>
<td>0-15</td>
<td>0-16</td>
<td>0-16</td>
<td>2.92307687</td>
<td>2.250000</td>
</tr>
<tr>
<td>0</td>
<td>13</td>
<td>0-13</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.06666667</td>
<td>1.250000</td>
</tr>
<tr>
<td>0</td>
<td>14</td>
<td>0-14</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.17142859</td>
<td>2.500000</td>
</tr>
<tr>
<td>0</td>
<td>15</td>
<td>0-15</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.20000002</td>
<td>1.250000</td>
</tr>
<tr>
<td>0</td>
<td>16</td>
<td>0-16</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.30000001</td>
<td>1.000000</td>
</tr>
<tr>
<td>1</td>
<td>0</td>
<td>1-0</td>
<td>len</td>
<td>19.7</td>
<td>1-1</td>
<td>1-2</td>
<td>1-2</td>
<td>3.51329851</td>
<td>14.695991</td>
</tr>
<tr>
<td>1</td>
<td>1</td>
<td>1-1</td>
<td>dose</td>
<td>1.0</td>
<td>1-3</td>
<td>1-4</td>
<td>1-4</td>
<td>1.63309026</td>
<td>7.308470</td>
</tr>
<tr>
<td>1</td>
<td>2</td>
<td>1-2</td>
<td>dose</td>
<td>2.0</td>
<td>1-5</td>
<td>1-6</td>
<td>1-6</td>
<td>1.65485406</td>
<td>7.387520</td>
</tr>
<tr>
<td>1</td>
<td>3</td>
<td>1-3</td>
<td>len</td>
<td>8.2</td>
<td>1-7</td>
<td>1-8</td>
<td>1-8</td>
<td>3.56799269</td>
<td>4.645680</td>
</tr>
<tr>
<td>1</td>
<td>4</td>
<td>1-4</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.28835031</td>
<td>2.662790</td>
</tr>
<tr>
<td>1</td>
<td>5</td>
<td>1-5</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.28835031</td>
<td>2.662790</td>
</tr>
<tr>
<td>1</td>
<td>6</td>
<td>1-6</td>
<td>len</td>
<td>26.7</td>
<td>1-9</td>
<td>1-10</td>
<td>1-10</td>
<td>0.22153124</td>
<td>4.724730</td>
</tr>
<tr>
<td>1</td>
<td>7</td>
<td>1-7</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.30163023</td>
<td>1.452431</td>
</tr>
<tr>
<td>1</td>
<td>8</td>
<td>1-8</td>
<td>len</td>
<td>11.2</td>
<td>1-11</td>
<td>1-12</td>
<td>1-12</td>
<td>0.25236940</td>
<td>3.193249</td>
</tr>
<tr>
<td>1</td>
<td>9</td>
<td>1-9</td>
<td>len</td>
<td>24.5</td>
<td>1-13</td>
<td>1-14</td>
<td>1-14</td>
<td>0.44972166</td>
<td>2.985818</td>
</tr>
<tr>
<td>1</td>
<td>10</td>
<td>1-10</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.05241550</td>
<td>1.738913</td>
</tr>
<tr>
<td>1</td>
<td>11</td>
<td>1-11</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.21860033</td>
<td>1.472866</td>
</tr>
<tr>
<td>1</td>
<td>12</td>
<td>1-12</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.03878851</td>
<td>1.720383</td>
</tr>
<tr>
<td>1</td>
<td>13</td>
<td>1-13</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>0.05559399</td>
<td>1.248612</td>
</tr>
<tr>
<td>1</td>
<td>14</td>
<td>1-14</td>
<td>Leaf</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>NA</td>
<td>-0.13160129</td>
<td>1.737206</td>
</tr>
</tbody>
</table>

# Other features

XGBoost supports many additional features on top of its traditional
gradient-boosting framework, including, among others:

-   Building decision tree models with characteristics such as
    per-feature monotonicity constraints or interaction constraints.
-   Calculating feature contributions in individual predictions.
-   Using custom objectives and custom evaluation metrics.
-   Fitting linear models.
-   Fitting models on GPUs and/or on data that doesn’t fit in RAM
    (“external memory”).

See the [online
documentation](https://xgboost.readthedocs.io/en/stable/index.html) -
particularly the [tutorials
section](https://xgboost.readthedocs.io/en/stable/tutorials/index.html) -
for a glimpse over further functionalities that XGBoost offers.

# The low-level interface

In addition to the `xgboost(x, y, ...)` function, XGBoost also provides
a lower-level interface for creating model objects through the function
`xgb.train()`, which resembles the same `xgb.train` functions in other
language bindings of XGBoost.

This `xgb.train()` interface exposes additional functionalities (such as
user-supplied callbacks or external-memory data support) and performs
fewer data validations and castings compared to the `xgboost()` function
interface.

Some key differences between the two interfaces:

-   Unlike `xgboost()` which takes R objects such as `matrix` or
    `data.frame` as inputs, the function `xgb.train()` uses XGBoost’s
    own data container called “DMatrix”, which can be created from R
    objects through the function `xgb.DMatrix()`. Note that there are
    other “DMatrix” constructors too, such as “xgb.QuantileDMatrix()”,
    which might be more beneficial for some use-cases.
-   A “DMatrix” object may contain a mixture of features/covariates, the
    response variable, observation weights, base margins, among others;
    and unlike `xgboost()`, requires its inputs to have already been
    encoded into the representation that XGBoost uses behind the
    scenes - for example, while `xgboost()` may take a `factor` object
    as “y”, `xgb.DMatrix()` requires instead a binary response variable
    to be passed as a vector of zeros and ones.
-   Hyperparameters are passed as function arguments in `xgboost()`,
    while they are passed as a named list to `xgb.train()`.
-   The `xgb.train()` interface keeps less metadata about its inputs -
    for example, it will not add levels of factors as column names to
    estimated probabilities when calling `predict`.

Example usage of `xgb.train()`:

``` r
data("agaricus.train")
dmatrix <- xgb.DMatrix(
    data = agaricus.train$data,  # a sparse CSC matrix ('dgCMatrix')
    label = agaricus.train$label # zeros and ones
)
booster <- xgb.train(
    data = dmatrix,
    nrounds = 10,
    params = xgb.params(
        objective = "binary:logistic",
        nthread = 1,
        max_depth = 3
    )
)

data("agaricus.test")
dmatrix_test <- xgb.DMatrix(agaricus.test$data)
pred_prob <- predict(booster, dmatrix_test)
pred_raw <- predict(booster, dmatrix_test, outputmargin = TRUE)
```

Model objects produced by `xgb.train()` have class `xgb.Booster`, while
model objects produced by `xgboost()` have class `xgboost`, which is a
subclass of `xgb.Booster`. Their `predict` methods also take different
arguments - for example, `predict.xgboost` has a `type` parameter, while
`predict.xgb.Booster` controls this through binary arguments - but as
`xgboost` is a subclass of `xgb.Booster`, methods for `xgb.Booster` can
be called on `xgboost` objects if needed.

Utility functions in the XGBoost R package will work with both model
classes - for example:

``` r
xgb.importance(model)
xgb.importance(booster)
```

A data.table: 2 × 4

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Feature &lt;chr&gt;</th>
<th>Gain &lt;dbl&gt;</th>
<th>Cover &lt;dbl&gt;</th>
<th>Frequency &lt;dbl&gt;</th>
</tr>
</thead>
<tbody>
<tr>
<td>len</td>
<td>0.7444265</td>
<td>0.6830449</td>
<td>0.7333333</td>
</tr>
<tr>
<td>dose</td>
<td>0.2555735</td>
<td>0.3169551</td>
<td>0.2666667</td>
</tr>
</tbody>
</table>

A data.table: 15 × 4

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Feature &lt;chr&gt;</th>
<th>Gain &lt;dbl&gt;</th>
<th>Cover &lt;dbl&gt;</th>
<th>Frequency &lt;dbl&gt;</th>
</tr>
</thead>
<tbody>
<tr>
<td>odor=none</td>
<td>0.6083687503</td>
<td>0.3459792871</td>
<td>0.16949153</td>
</tr>
<tr>
<td>stalk-root=club</td>
<td>0.0959684807</td>
<td>0.0695742744</td>
<td>0.03389831</td>
</tr>
<tr>
<td>odor=anise</td>
<td>0.0645662853</td>
<td>0.0777761744</td>
<td>0.10169492</td>
</tr>
<tr>
<td>odor=almond</td>
<td>0.0542574659</td>
<td>0.0865120182</td>
<td>0.10169492</td>
</tr>
<tr>
<td>bruises?=bruises</td>
<td>0.0532525762</td>
<td>0.0535293301</td>
<td>0.06779661</td>
</tr>
<tr>
<td>stalk-root=rooted</td>
<td>0.0471992509</td>
<td>0.0610565707</td>
<td>0.03389831</td>
</tr>
<tr>
<td>spore-print-color=green</td>
<td>0.0326096192</td>
<td>0.1418126308</td>
<td>0.16949153</td>
</tr>
<tr>
<td>odor=foul</td>
<td>0.0153302980</td>
<td>0.0103517575</td>
<td>0.01694915</td>
</tr>
<tr>
<td>stalk-surface-below-ring=scaly</td>
<td>0.0126892940</td>
<td>0.0914230316</td>
<td>0.08474576</td>
</tr>
<tr>
<td>gill-size=broad</td>
<td>0.0066973198</td>
<td>0.0345993858</td>
<td>0.10169492</td>
</tr>
<tr>
<td>odor=pungent</td>
<td>0.0027091458</td>
<td>0.0032193586</td>
<td>0.01694915</td>
</tr>
<tr>
<td>population=clustered</td>
<td>0.0025750464</td>
<td>0.0015616374</td>
<td>0.03389831</td>
</tr>
<tr>
<td>stalk-color-below-ring=yellow</td>
<td>0.0016913567</td>
<td>0.0173903519</td>
<td>0.01694915</td>
</tr>
<tr>
<td>spore-print-color=white</td>
<td>0.0012798160</td>
<td>0.0008031107</td>
<td>0.01694915</td>
</tr>
<tr>
<td>gill-spacing=close</td>
<td>0.0008052948</td>
<td>0.0044110809</td>
<td>0.03389831</td>
</tr>
</tbody>
</table>

While `xgboost()` aims to provide a user-friendly interface, there are
still many situations where one should prefer the `xgb.train()`
interface - for example:

-   For latency-sensitive applications (e.g. when serving models in real
    time), `xgb.train()` will have a speed advantage, as it performs
    fewer validations, conversions, and post-processings with metadata.
-   If you are developing an R package that depends on XGBoost,
    `xgb.train()` will provide a more stable interface (less subject to
    changes) and will have lower time/memory overhead.
-   If you need functionalities that are not exposed by the `xgboost()`
    interface - for example, if your dataset does not fit into the
    computer’s RAM, it’s still possible to construct a DMatrix from it
    if the data is loaded in batches through `xgb.ExtMemDMatrix()`.
