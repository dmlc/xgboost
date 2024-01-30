"""Xgboost pyspark integration submodule for params."""

from typing import Dict

# pylint: disable=too-few-public-methods
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params


class HasArbitraryParamsDict(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the **kwargs parts of the XGBoost
    input.
    """

    arbitrary_params_dict: "Param[Dict]" = Param(
        Params._dummy(),
        "arbitrary_params_dict",
        "arbitrary_params_dict This parameter holds all of the additional parameters which are "
        "not exposed as the XGBoost Spark estimator params but can be recognized by "
        "underlying XGBoost library. It is stored as a dictionary.",
    )


class HasBaseMarginCol(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the base margin column part of XGboost.
    """

    base_margin_col = Param(
        Params._dummy(),
        "base_margin_col",
        "This stores the name for the column of the base margin",
        typeConverter=TypeConverters.toString,
    )


class HasFeaturesCols(Params):
    """
    Mixin for param features_cols: a list of feature column names.
    This parameter is taken effect only when use_gpu is enabled.
    """

    features_cols = Param(
        Params._dummy(),
        "features_cols",
        "feature column names.",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(features_cols=[])


class HasEnableSparseDataOptim(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the boolean config of enabling sparse data optimization.
    """

    enable_sparse_data_optim = Param(
        Params._dummy(),
        "enable_sparse_data_optim",
        "This stores the boolean config of enabling sparse data optimization, if enabled, "
        "Xgboost DMatrix object will be constructed from sparse matrix instead of "
        "dense matrix. This config is disabled by default. If most of examples in your "
        "training dataset contains sparse features, we suggest to enable this config.",
        typeConverter=TypeConverters.toBoolean,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(enable_sparse_data_optim=False)


class HasQueryIdCol(Params):
    """
    Mixin for param qid_col: query id column name.
    """

    qid_col = Param(
        Params._dummy(),
        "qid_col",
        "query id column name",
        typeConverter=TypeConverters.toString,
    )


class HasContribPredictionCol(Params):
    """
    Mixin for param pred_contrib_col: contribution prediction column name.

    Output is a 3-dim array, with (rows, groups, columns + 1) for classification case.
    Else, it can be a 2 dimension for regression case.
    """

    pred_contrib_col: "Param[str]" = Param(
        Params._dummy(),
        "pred_contrib_col",
        "feature contributions to individual predictions.",
        typeConverter=TypeConverters.toString,
    )
