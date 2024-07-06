###################################
Advanced Usage of Custom Objectives
###################################

**Contents**

.. contents::
  :backlinks: none
  :local:

********
Overview
********

XGBoost allows optimizing custom user-defined functions based on
gradients and Hessians provided by the user for the desired objective function.

In order for a custom objective to work as intended:

- The function to optimize must be smooth and twice differentiable.
- The function must be additive with respect to rows / observations,
  such as a likelihood function with i.i.d. assumptions.
- The range of the scores for the function must be unbounded
  (i.e. it should not work exclusively with positive numbers, for example).
- The function must be convex. Note that, if the Hessian has negative
  values, they will be clipped, which will likely result in a model
  that does not fit the function well.
- For multi-output objectives, there should not be dependencies between
  different targets (i.e. Hessian should be diagonal for each row).


Some of these limitations can nevertheless be worked around by foregoing
the true Hessian of the function, using something else instead such as an
approximation with better properties - convergence might be slower when
not using the true Hessian of a function, but many theoretical guarantees
should still hold and result in usable models. For example, XGBoost's
internal implementation of multionomial logistic regression uses an upper
bound on the Hessian with diagonal structure instead of the true Hessian
which is a full square matrix for each row in the data.

This tutorial provides some suggestions for use-cases that do not perfectly
fit the criteria outlined above, by showing how to solve a Dirichlet regression
parameterized by concentrations.

A Dirichlet regression model poses certain challenges for XGBoost:

- Concentration parameters must be positive. An easy way to achieve this is
  by applying an 'exp' transform on raw unbounded values, but in such case
  the objective becomes non-convex. Furthermore, note that this function is
  not in the exponential family, unlike typical distributions used for GLM
  models.
- The Hessian has dependencies between targets - that is, for a Dirichlet
  distribution with 'k' parameters, each row will have a full Hessian matrix
  of dimensions ``[k, k]``.
- An optimal intercept for this type of model would involve a vector of
  values rather than the same value for every target.

In order to use this type of model as a custom objetive:

- It's possible to use the expected Hessian (a.k.a. the Fisher information
  matrix or expected information) instead of the true Hessian. The expected
  Hessian is always positive semi-definite for an additive likelihood, even
  if the true Hessian isn't.
- It's possible to use an upper bound on the expected Hessian with a diagonal
  structure, such that a second-order approximation under this diagonal
  bound would always yield greater or equal function values than under the
  non-diagonal expected Hessian.
- Since the ``base_score`` parameter that XGBoost uses for an intercept is
  limited to a scalar, one can use the ``base_margin`` functionality instead,
  but note that using it requires a bit more effort.

*****************************
Dirichlet Regression Formulae
*****************************

The Dirichlet distribution is a generalization of the Beta distribution to
multiple dimensions. It models proportions data in which the values sum to
1, and is typically used as part of composite models (e.g. Dirichlet-multinomial)
or as a prior in Bayesian models, but it also can be used on its own for
proportions data for example.

Its likelihood for a given observation with values ``y`` and a given prediction ``x``
is given as follows:

.. math::
    L(\mathbf{y} | \mathbf{x}) = \frac{1}{\beta(\mathbf{x})} \prod_{i=1}^k y_i^{x_i - 1}

Where:

.. math::
  \beta(\mathbf{x}) = \frac{ \prod_{i=1}^k \Gamma(x_i) }{\Gamma( \sum_{i=1}^k x_i )}


In this case, we want to optimize the negative of the log-likelihood summed across rows.
The resulting function, gradient and Hessian could be implemented as follows:

.. code-block:: python
    :caption: Python

    import numpy as np
    from scipy.special import loggamma, psi as digamma, polygamma
    trigamma = lambda x: polygamma(1, x)

    def dirichlet_fun(pred: np.ndarray, Y: np.ndarray) -> float:
        epred = np.exp(pred)
        sum_epred = np.sum(epred, axis=1, keepdims=True)
        return (
            loggamma(epred).sum()
            - loggamma(sum_epred).sum()
            - np.sum(np.log(Y) * (epred - 1))
        )
    def dirichlet_grad(pred: np.ndarray, Y: np.ndarray) -> np.ndarray:
        epred = np.exp(pred)
        return epred * (
            digamma(epred)
            - digamma(np.sum(epred, axis=1, keepdims=True))
            - np.log(Y)
        )
    def dirichlet_hess(pred: np.ndarray, Y: np.ndarray) -> np.ndarray:
        epred = np.exp(pred)
        grad = dirichlet_grad(pred, Y)
        k = Y.shape[1]
        H = np.empty((pred.shape[0], k, k))
        for row in range(pred.shape[0]):
            H[row, :, :] = (
                - trigamma(epred[row].sum()) * np.outer(epred[row], epred[row])
                + np.diag(grad[row] + trigamma(epred[row]) * epred[row] ** 2)
            )
        return H

.. code-block:: r
    :caption: R

    softmax <- function(x) {
        max.x <- max(x)
        e <- exp(x - max.x)
        return(e / sum(e))
    }

    dirichlet.fun <- function(pred, y) {
        epred <- exp(pred)
        sum_epred <- rowSums(epred)
        return(
            sum(lgamma(epred))
            - sum(lgamma(sum_epred))
            - sum(log(y) * (epred - 1))
        )
    }

    dirichlet.grad <- function(pred, y) {
        epred <- exp(pred)
        return(
            epred * (
                digamma(epred)
                - digamma(rowSums(epred))
                - log(y)
            )
        )
    }

    dirichlet.hess <- function(pred, y) {
        epred <- exp(pred)
        grad <- dirichlet.grad(pred, y)
        k <- ncol(y)
        H <- array(dim = c(nrow(y), k, k))
        for (row in seq_len(nrow(y))) {
            H[row, , ] <- (
                - trigamma(sum(epred[row,])) * tcrossprod(epred[row,])
                + diag(grad[row,] + trigamma(epred[row,]) * epred[row,]^2)
            )
        }
        return(H)
    }


Convince yourself that the implementation is correct:

.. code-block:: python
    :caption: Python

    from math import isclose
    from scipy import stats
    from scipy.optimize import check_grad
    from scipy.special import softmax

    def gen_random_dirichlet(rng: np.random.Generator, m: int, k: int):
        alpha = np.exp(rng.standard_normal(size=k))
        return rng.dirichlet(alpha, size=m)
    
    def test_dirichlet_fun_grad_hess():
        k = 3
        m = 10
        rng = np.random.default_rng(seed=123)
        Y = gen_random_dirichlet(rng, m, k)
        x0 = rng.standard_normal(size=k)
        for row in range(Y.shape[0]):
            fun_row = dirichlet_fun(x0.reshape((1,-1)), Y[[row]])
            ref_logpdf = stats.dirichlet.logpdf(
                Y[row] / Y[row].sum(), # <- avoid roundoff error
                np.exp(x0),
            )
            assert isclose(fun_row, -ref_logpdf)

            gdiff = check_grad(
                lambda pred: dirichlet_fun(pred.reshape((1,-1)), Y[[row]]),
                lambda pred: dirichlet_grad(pred.reshape((1,-1)), Y[[row]]),
                x0
            )
            assert gdiff <= 1e-6

            H_numeric = np.empty((k,k))
            eps = 1e-7
            for ii in range(k):
                x0_plus_eps = x0.reshape((1,-1)).copy()
                x0_plus_eps[0,ii] += eps
                for jj in range(k):
                    H_numeric[ii, jj] = (
                        dirichlet_grad(x0_plus_eps, Y[[row]])[0][jj]
                        - dirichlet_grad(x0.reshape((1,-1)), Y[[row]])[0][jj]
                    ) / eps
            H = dirichlet_hess(x0.reshape((1,-1)), Y[[row]])[0]
            np.testing.assert_almost_equal(H, H_numeric, decimal=6)
    test_dirichlet_fun_grad_hess()


.. code-block:: r
    :caption: R

    library(DirichletReg)
    library(testthat)

    test_that("dirichlet formulae", {
        k <- 3L
        m <- 10L
        set.seed(123)
        alpha <- exp(rnorm(k))
        y <- rdirichlet(m, alpha)
        x0 <- rnorm(k)
        
        for (row in seq_len(m)) {
            logpdf <- dirichlet.fun(matrix(x0, nrow=1), y[row,,drop=F])
            ref_logpdf <- ddirichlet(y[row,,drop=F], exp(x0), log = T)
            expect_equal(logpdf, -ref_logpdf)
            
            eps <- 1e-7
            grad_num <- numeric(k)
            for (col in seq_len(k)) {
                xplus <- x0
                xplus[col] <- x0[col] + eps
                grad_num[col] <- (
                    dirichlet.fun(matrix(xplus, nrow=1), y[row,,drop=F])
                    - dirichlet.fun(matrix(x0, nrow=1), y[row,,drop=F])
                ) / eps
            }
            
            grad <- dirichlet.grad(matrix(x0, nrow=1), y[row,,drop=F])
            expect_equal(grad |> as.vector(), grad_num, tolerance=1e-6)
            
            H_numeric <- array(dim=c(k, k))
            for (ii in seq_len(k)) {
                xplus <- x0
                xplus[ii] <- x0[ii] + eps
                for (jj in seq_len(k)) {
                    H_numeric[ii, jj] <- (
                        dirichlet.grad(matrix(xplus, nrow=1), y[row,,drop=F])[1, jj]
                        - grad[1L, jj]
                    ) / eps
                }
            }
            
            H <- dirichlet.hess(matrix(xplus, nrow=1), y[row,,drop=F])
            expect_equal(H[1,,], H_numeric, tolerance=1e-6)
        }
    })

******************************************
Dirichlet Regression as Objective Function
******************************************

As mentioned earlier, the Hessian of this function is problematic for
XGBoost: it can have a negative determinant, and might even have negative
values in the diagonal, which is problematic for optimization methods - in
XGBoost, those values would be clipped and the resulting model might not
end up producing sensible predictions.

A potential workaround is to use the expected Hessian instead - that is,
the expected outer product of the gradient if the response variable were
distributed according to what is predicted. See the Wikipedia article
for more information:

`<https://en.wikipedia.org/wiki/Fisher_information>`_

In general, for objective functions in the exponential family, this is easy
to obtain from the gradient of the link function and the variance of the
probability distribution, but for other functions in general, it might
involve other types of calculations (e.g. covariances and covariances of
logarithms for Dirichlet).

It nevertheless results in a form very similar to the Hessian. One can also
see from the differences here that, at an optimal point (gradient being zero),
the expected and true Hessian for Dirichlet will match, which is a nice
property for optimization (i.e. the Hessian will be positive at a stationary
point, which means it will be a minimum rather than a maximum or saddle point).

.. code-block:: python
    :caption: Python

    def dirichlet_expected_hess(pred: np.ndarray) -> np.ndarray:
        epred = np.exp(pred)
        k = pred.shape[1]
        Ehess = np.empty((pred.shape[0], k, k))
        for row in range(pred.shape[0]):
            Ehess[row, :, :] = (
                - trigamma(epred[row].sum()) * np.outer(epred[row], epred[row])
                + np.diag(trigamma(epred[row]) * epred[row] ** 2)
            )
        return Ehess
    def test_dirichlet_expected_hess():
        k = 3
        rng = np.random.default_rng(seed=123)
        x0 = rng.standard_normal(size=k)
        y_sample = rng.dirichlet(np.exp(x0), size=5_000_000)
        x_broadcast = np.broadcast_to(x0, (y_sample.shape[0], k))
        g_sample = dirichlet_grad(x_broadcast, y_sample)
        ref = (g_sample.T @ g_sample) / y_sample.shape[0]
        Ehess = dirichlet_expected_hess(x0.reshape((1,-1)))[0]
        np.testing.assert_almost_equal(Ehess, ref, decimal=2)
    test_dirichlet_expected_hess()

.. code-block:: r
    :caption: R

    dirichlet.expected.hess <- function(pred) {
        epred <- exp(pred)
        k <- ncol(pred)
        H <- array(dim = c(nrow(pred), k, k))
        for (row in seq_len(nrow(pred))) {
            H[row, , ] <- (
                - trigamma(sum(epred[row,])) * tcrossprod(epred[row,])
                + diag(trigamma(epred[row,]) * epred[row,]^2)
            )
        }
        return(H)
    }

    test_that("expected hess", {
        k <- 3L
        set.seed(123)
        x0 <- rnorm(k)
        alpha <- exp(x0)
        n.samples <- 5e6
        y.samples <- rdirichlet(n.samples, alpha)
        
        x.broadcast <- rep(x0, n.samples) |> matrix(ncol=k, byrow=T)
        grad.samples <- dirichlet.grad(x.broadcast, y.samples)
        ref <- crossprod(grad.samples) / n.samples
        Ehess <- dirichlet.expected.hess(matrix(x0, nrow=1))
        expect_equal(Ehess[1,,], ref, tolerance=1e-2)
    })

But note that this is still not usable for XGBoost, since the expected
Hessian, just like the true Hessian, has shape ``[nrows, k, k]``, while
XGBoost requires something with shape ``[k, k]``.

One may use the diagonal of the expected Hessian for each row, but it's
possible to do better: one can use instead an upper bound with diagonal
structure, since it should lead to better convergence properties, just like
for other Hessian-based optimization methods.

In the absence of any obvious way of obtaining an upper bound, a possibility
here is to construct such a bound numerically based directly on the definition
of a diagonally dominant matrix:

`<https://en.wikipedia.org/wiki/Diagonally_dominant_matrix>`_

That is: take the absolute value of the expected Hessian for each row of the data,
and sum by rows of the ``[k, k]``-shaped Hessian for that row in the data:

.. code-block:: python
    :caption: Python

    def dirichlet_diag_upper_bound_expected_hess(
        pred: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        Ehess = dirichlet_expected_hess(pred)
        diag_bound_Ehess = np.empty((pred.shape[0], Y.shape[1]))
        for row in range(pred.shape[0]):
            diag_bound_Ehess[row, :] = np.abs(Ehess[row, :, :]).sum(axis=1)
        return diag_bound_Ehess

.. code-block:: r
    :caption: R

    dirichlet.diag.upper.bound.expected.hess <- function(pred, y) {
        Ehess <- dirichlet.expected.hess(pred)
        diag.bound.Ehess <- array(dim=dim(pred))
        for (row in seq_len(nrow(pred))) {
            diag.bound.Ehess[row,] <- abs(Ehess[row,,]) |> rowSums()
        }
        return(diag.bound.Ehess)
    }

(*note: the calculation can be made more efficiently than what is shown here
by not calculating the full matrix, and in R, by making the rows be the last
dimension and transposing after the fact*)

With all these pieces in place, one can now frame this model into the format
required for XGBoost's custom objectives:

.. code-block:: python
    :caption: Python

    import xgboost as xgb
    from typing import Tuple

    def dirichlet_xgb_objective(
        pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        Y = dtrain.get_label().reshape(pred.shape)
        return (
            dirichlet_grad(pred, Y),
            dirichlet_diag_upper_bound_expected_hess(pred, Y),
        )

.. code-block:: r
    :caption: R

    library(xgboost)
    
    dirichlet.xgb.objective <- function(pred, dtrain) {
        y <- getinfo(dtrain, "label")
        return(
            list(
                grad = dirichlet.grad(pred, y),
                hess = dirichlet.diag.upper.bound.expected.hess(pred, y)
            )
        )
    }

And for an evaluation metric monitoring based on the Dirichlet log-likelihood:

.. code-block:: python
    :caption: Python

    def dirichlet_eval_metric(
        pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> Tuple[str, float]:
        Y = dtrain.get_label().reshape(pred.shape)
        return "dirichlet_ll", dirichlet_fun(pred, Y)

.. code-block:: r
    :caption: R

    dirichlet.eval.metric <- function(pred, dtrain) {
        y <- getinfo(dtrain, "label")
        ll <- dirichlet.fun(pred, y)
        return(
            list(
                metric = "dirichlet_ll",
                value = ll
            )
        )
    }

*****************
Practical Example
*****************

A good source for test datasets for proportions data is the R package ``DirichletReg``:

`<https://cran.r-project.org/package=DirichletReg>`_

For this example, we'll now use the Arctic Lake dataset
(Aitchison, J. (2003). The Statistical Analysis of Compositional Data. The Blackburn Press, Caldwell, NJ.),
taken from the ``DirichletReg`` R package, which consists of 39 rows with one predictor variable 'depth'
and a three-valued response variable denoting the sediment composition of the measurements in this arctic
lake (sand, silt, clay).

The data:

.. code-block:: python
    :caption: Python
    
    # depth
    X = np.array([
        10.4,11.7,12.8,13,15.7,16.3,18,18.7,20.7,22.1,
        22.4,24.4,25.8,32.5,33.6,36.8,37.8,36.9,42.2,47,
        47.1,48.4,49.4,49.5,59.2,60.1,61.7,62.4,69.3,73.6,
        74.4,78.5,82.9,87.7,88.1,90.4,90.6,97.7,103.7,
    ]).reshape((-1,1))
    # sand, silt, clay
    Y = np.array([
        [0.775,0.195,0.03], [0.719,0.249,0.032], [0.507,0.361,0.132],
        [0.522,0.409,0.066], [0.7,0.265,0.035], [0.665,0.322,0.013],
        [0.431,0.553,0.016], [0.534,0.368,0.098], [0.155,0.544,0.301],
        [0.317,0.415,0.268], [0.657,0.278,0.065], [0.704,0.29,0.006],
        [0.174,0.536,0.29], [0.106,0.698,0.196], [0.382,0.431,0.187],
        [0.108,0.527,0.365], [0.184,0.507,0.309], [0.046,0.474,0.48],
        [0.156,0.504,0.34], [0.319,0.451,0.23], [0.095,0.535,0.37],
        [0.171,0.48,0.349], [0.105,0.554,0.341], [0.048,0.547,0.41],
        [0.026,0.452,0.522], [0.114,0.527,0.359], [0.067,0.469,0.464],
        [0.069,0.497,0.434], [0.04,0.449,0.511], [0.074,0.516,0.409],
        [0.048,0.495,0.457], [0.045,0.485,0.47], [0.066,0.521,0.413],
        [0.067,0.473,0.459], [0.074,0.456,0.469], [0.06,0.489,0.451],
        [0.063,0.538,0.399], [0.025,0.48,0.495], [0.02,0.478,0.502],
    ])

.. code-block:: r
    :caption: R

    data("ArcticLake", package="DirichletReg")
    x <- ArcticLake[, c("depth"), drop=F]
    y <- ArcticLake[, c("sand", "silt", "clay")] |> as.matrix()

Fitting an XGBoost model and making predictions:

.. code-block:: python
    :caption: Python
    
    from typing import Dict, List
    
    dtrain = xgb.DMatrix(X, label=Y)
    results: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params={
            "tree_method": "hist",
            "num_target": Y.shape[1],
            "base_score": 0,
            "disable_default_eval_metric": True,
            "max_depth": 3,
            "seed": 123,
        },
        dtrain=dtrain,
        num_boost_round=10,
        obj=dirichlet_xgb_objective,
        evals=[(dtrain, "Train")],
        evals_result=results,
        custom_metric=dirichlet_eval_metric,
    )
    yhat = softmax(booster.inplace_predict(X), axis=1)

.. code-block:: r
    :caption: R

    dtrain <- xgb.DMatrix(x, y)
    booster <- xgb.train(
        params = list(
            tree_method="hist",
            num_target=ncol(y),
            base_score=0,
            disable_default_eval_metric=TRUE,
            max_depth=3,
            seed=123
        ),
        data = dtrain,
        nrounds = 10,
        obj = dirichlet.xgb.objective,
        evals = list(Train=dtrain),
        eval_metric = dirichlet.eval.metric
    )
    raw.pred <- predict(booster, x, reshape=TRUE)
    yhat <- apply(raw.pred, 1, softmax) |> t()


Should produce an evaluation log as follows (note: the function is decreasing as
expected - but unlike other objectives, the minimum value here can reach below zero):

.. code-block:: none

    [0] Train-dirichlet_ll:-40.25009
    [1] Train-dirichlet_ll:-47.69122
    [2] Train-dirichlet_ll:-52.64620
    [3] Train-dirichlet_ll:-56.36977
    [4] Train-dirichlet_ll:-59.33048
    [5] Train-dirichlet_ll:-61.93359
    [6] Train-dirichlet_ll:-64.17280
    [7] Train-dirichlet_ll:-66.29709
    [8] Train-dirichlet_ll:-68.21001
    [9] Train-dirichlet_ll:-70.03442

One can confirm that the obtained ``yhat`` resembles the actual concentrations
to a large degree, beyond what would be expected from random predictions by a
simple look at both ``yhat`` and ``Y``.

For better results, one might want to add an intercept. XGBoost only
allows using scalars for intercepts, but for a vector-valued model,
the optimal intercept should also have vector form.

This can be done by supplying ``base_margin`` instead - unlike the
intercept, one must specifically supply values for every row here,
and said ``base_margin`` must be supplied again at the moment of making
predictions (i.e. does not get added automatically like ``base_score``
does).

For the case of a Dirichlet model, the optimal intercept can be obtained
efficiently using a general solver (e.g. SciPy's Newton solver) with
dedicated likelihood, gradient and Hessian functions for just the intercept part.
Further, note that if one frames it instead as bounded optimization without
applying 'exp' transform to the concentrations, it becomes instead a convex
problem, for which the true Hessian can be used without issues in other
classes of solvers.

For simplicity, this example will nevertheless reuse the same likelihood
and gradient functions that were defined earlier alongside with SciPy's / R's
L-BFGS solver to obtain the optimal vector-valued intercept:

.. code-block:: python
    :caption: Python

    from scipy.optimize import minimize

    def get_optimal_intercepts(Y: np.ndarray) -> np.ndarray:
        k = Y.shape[1]
        res = minimize(
            fun=lambda pred: dirichlet_fun(
                np.broadcast_to(pred, (Y.shape[0], k)),
                Y
            ),
            x0=np.zeros(k),
            jac=lambda pred: dirichlet_grad(
                np.broadcast_to(pred, (Y.shape[0], k)),
                Y
            ).sum(axis=0)
        )
        return res["x"]
    intercepts = get_optimal_intercepts(Y)

.. code-block:: r
    :caption: R

    get.optimal.intercepts <- function(y) {
        k <- ncol(y)
        broadcast.vec <- function(x) rep(x, nrow(y)) |> matrix(ncol=k, byrow=T)
        res <- optim(
            par = numeric(k),
            fn = function(x) dirichlet.fun(broadcast.vec(x), y),
            gr = function(x) dirichlet.grad(broadcast.vec(x), y) |> colSums(),
            method = "L-BFGS-B"
        )
        return(res$par)
    }
    intercepts <- get.optimal.intercepts(y)


Now fitting a model again, this time with the intercept:

.. code-block:: python
    :caption: Python

    base_margin = np.broadcast_to(intercepts, Y.shape)
    dtrain_w_intercept = xgb.DMatrix(X, label=Y, base_margin=base_margin)
    results: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params={
            "tree_method": "hist",
            "num_target": Y.shape[1],
            "base_score": 0,
            "disable_default_eval_metric": True,
            "max_depth": 3,
            "seed": 123,
        },
        dtrain=dtrain_w_intercept,
        num_boost_round=10,
        obj=dirichlet_xgb_objective,
        evals=[(dtrain, "Train")],
        evals_result=results,
        custom_metric=dirichlet_eval_metric,
    )
    yhat = softmax(
        booster.predict(
            xgb.DMatrix(X, base_margin=base_margin)
        ),
        axis=1
    )

.. code-block:: r
    :caption: R

    base.margin <- rep(intercepts, nrow(y)) |> matrix(nrow=nrow(y), byrow=T)
    dtrain <- xgb.DMatrix(x, y, base_margin=base.margin)
    booster <- xgb.train(
        params = list(
            tree_method="hist",
            num_target=ncol(y),
            base_score=0,
            disable_default_eval_metric=TRUE,
            max_depth=3,
            seed=123
        ),
        data = dtrain,
        nrounds = 10,
        obj = dirichlet.xgb.objective,
        evals = list(Train=dtrain),
        eval_metric = dirichlet.eval.metric
    )
    raw.pred <- predict(
        booster,
        x,
        base_margin=base.margin,
        reshape=TRUE
    )
    yhat <- apply(raw.pred, 1, softmax) |> t()

.. code-block:: none

    [0] Train-dirichlet_ll:-37.01861
    [1] Train-dirichlet_ll:-42.86120
    [2] Train-dirichlet_ll:-46.55133
    [3] Train-dirichlet_ll:-49.15111
    [4] Train-dirichlet_ll:-51.02638
    [5] Train-dirichlet_ll:-52.53880
    [6] Train-dirichlet_ll:-53.77409
    [7] Train-dirichlet_ll:-54.88851
    [8] Train-dirichlet_ll:-55.95961
    [9] Train-dirichlet_ll:-56.95497

For this small example problem, predictions should be very similar between the
two and the version without intercepts achieved a lower objective function in the
training data (for the Python version at least), but for more serious usage with
real-world data, one is likely to observe better results when adding the intercepts.
