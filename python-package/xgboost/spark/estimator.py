# type: ignore
"""Xgboost pyspark integration submodule for estimator API."""
# pylint: disable=too-many-ancestors
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRegressor
from .core import (
    _SparkXGBEstimator,
    SparkXGBClassifierModel,
    SparkXGBRegressorModel,
    _set_pyspark_xgb_cls_param_attrs,
)


class SparkXGBRegressor(_SparkXGBEstimator):
    """
    SparkXGBRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    SparkXGBRegressor automatically supports most of the parameters in
    `xgboost.XGBRegressor` constructor and most of the parameters used in
    `xgboost.XGBRegressor` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBRegressor>`_ for details).

    SparkXGBRegressor doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    SparkXGBRegressor doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBRegressor doesn't support `validate_features` and `output_margin` param.

    callbacks:
        The export and import of the callback functions are at best effort.
        For details, see :py:attr:`xgboost.spark.SparkXGBRegressor.callbacks` param doc.
    validation_indicator_col
        For params related to `xgboost.XGBRegressor` training
        with evaluation dataset's supervision, set
        :py:attr:`xgboost.spark.SparkXGBRegressor.validation_indicator_col`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    weight_col:
        To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.SparkXGBRegressor.weight_col` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    xgb_model:
        Set the value to be the instance returned by
        :func:`xgboost.spark.SparkXGBRegressorModel.get_booster`.
    num_workers:
        Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one spark task.
    use_gpu:
        Boolean that specifies whether the executors are running on GPU
        instances.
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBRegressor.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBRegressor` fit method. Note: this isn't available for distributed
        training.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from xgboost.spark import SparkXGBRegressor
    >>> from pyspark.ml.linalg import Vectors
    >>> df_train = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    ...     (Vectors.dense(4.0, 5.0, 6.0), 2, True, 1.0),
    ...     (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 3, True, 2.0),
    ... ], ["features", "label", "isVal", "weight"])
    >>> df_test = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), ),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), )
    ... ], ["features"])
    >>> xgb_regressor = SparkXGBRegressor(max_depth=5, missing=0.0,
    ... validation_indicator_col='isVal', weight_col='weight',
    ... early_stopping_rounds=1, eval_metric='rmse')
    >>> xgb_reg_model = xgb_regressor.fit(df_train)
    >>> xgb_reg_model.transform(df_test)

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.setParams(**kwargs)

    @classmethod
    def _xgb_cls(cls):
        return XGBRegressor

    @classmethod
    def _pyspark_model_cls(cls):
        return SparkXGBRegressorModel


_set_pyspark_xgb_cls_param_attrs(SparkXGBRegressor, SparkXGBRegressorModel)


class SparkXGBClassifier(_SparkXGBEstimator, HasProbabilityCol, HasRawPredictionCol):
    """
    SparkXGBClassifier is a PySpark ML estimator. It implements the XGBoost classification
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    SparkXGBClassifier automatically supports most of the parameters in
    `xgboost.XGBClassifier` constructor and most of the parameters used in
    `xgboost.XGBClassifier` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBClassifier>`_ for details).

    SparkXGBClassifier doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    SparkXGBClassifier doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBClassifier doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `raw_prediction_col` param doc below for more details.

    SparkXGBClassifier doesn't support `validate_features` and `output_margin` param.

    Parameters
    ----------
    callbacks:
        The export and import of the callback functions are at best effort. For
        details, see :py:attr:`xgboost.spark.SparkXGBClassifier.callbacks` param doc.
    raw_prediction_col:
        The `output_margin=True` is implicitly supported by the
        `rawPredictionCol` output column, which is always returned with the predicted margin
        values.
    validation_indicator_col:
        For params related to `xgboost.XGBClassifier` training with
        evaluation dataset's supervision,
        set :py:attr:`xgboost.spark.SparkXGBClassifier.validation_indicator_col`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    weight_col:
        To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.SparkXGBClassifier.weight_col` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    xgb_model:
        Set the value to be the instance returned by
        :func:`xgboost.spark.SparkXGBClassifierModel.get_booster`.
    num_workers:
        Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one spark task.
    use_gpu:
        Boolean that specifies whether the executors are running on GPU
        instances.
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBClassifier.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBClassifier` fit method. Note: this isn't available for distributed
        training.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from xgboost.spark import SparkXGBClassifier
    >>> from pyspark.ml.linalg import Vectors
    >>> df_train = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    ...     (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
    ...     (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
    ... ], ["features", "label", "isVal", "weight"])
    >>> df_test = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), ),
    ... ], ["features"])
    >>> xgb_classifier = SparkXGBClassifier(max_depth=5, missing=0.0,
    ...     validation_indicator_col='isVal', weight_col='weight',
    ...     early_stopping_rounds=1, eval_metric='logloss')
    >>> xgb_clf_model = xgb_classifier.fit(df_train)
    >>> xgb_clf_model.transform(df_test).show()

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.setParams(**kwargs)

    @classmethod
    def _xgb_cls(cls):
        return XGBClassifier

    @classmethod
    def _pyspark_model_cls(cls):
        return SparkXGBClassifierModel


_set_pyspark_xgb_cls_param_attrs(SparkXGBClassifier, SparkXGBClassifierModel)
