# type: ignore
"""Xgboost pyspark integration submodule for estimator API."""
# pylint: disable=too-many-ancestors
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol

from xgboost import XGBClassifier, XGBRanker, XGBRegressor

from .core import (
    SparkXGBClassifierModel,
    SparkXGBRankerModel,
    SparkXGBRegressorModel,
    _set_pyspark_xgb_cls_param_attrs,
    _SparkXGBEstimator,
)


class SparkXGBRegressor(_SparkXGBEstimator):
    """
    SparkXGBRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBRegressor automatically supports most of the parameters in
    `xgboost.XGBRegressor` constructor and most of the parameters used in
    :py:class:`xgboost.XGBRegressor` fit and predict method.

    SparkXGBRegressor doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    SparkXGBRegressor doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBRegressor doesn't support `validate_features` and `output_margin` param.

    SparkXGBRegressor doesn't support setting `nthread` xgboost param, instead, the `nthread`
    param for each xgboost worker will be set equal to `spark.task.cpus` config value.

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

    Examples
    --------

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

    def _validate_params(self):
        super()._validate_params()
        if self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost regressor estimator does not support `qid_col` param."
            )


_set_pyspark_xgb_cls_param_attrs(SparkXGBRegressor, SparkXGBRegressorModel)


class SparkXGBClassifier(_SparkXGBEstimator, HasProbabilityCol, HasRawPredictionCol):
    """SparkXGBClassifier is a PySpark ML estimator. It implements the XGBoost
    classification algorithm based on XGBoost python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBClassifier automatically supports most of the parameters in
    `xgboost.XGBClassifier` constructor and most of the parameters used in
    :py:class:`xgboost.XGBClassifier` fit and predict method.

    SparkXGBClassifier doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    SparkXGBClassifier doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBClassifier doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `raw_prediction_col` param doc below for more details.

    SparkXGBClassifier doesn't support `validate_features` and `output_margin` param.

    SparkXGBClassifier doesn't support setting `nthread` xgboost param, instead, the `nthread`
    param for each xgboost worker will be set equal to `spark.task.cpus` config value.


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

    Examples
    --------

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
        # The default 'objective' param value comes from sklearn `XGBClassifier` ctor,
        # but in pyspark we will automatically set objective param depending on
        # binary or multinomial input dataset, and we need to remove the fixed default
        # param value as well to avoid causing ambiguity.
        self._setDefault(objective=None)
        self.setParams(**kwargs)

    @classmethod
    def _xgb_cls(cls):
        return XGBClassifier

    @classmethod
    def _pyspark_model_cls(cls):
        return SparkXGBClassifierModel

    def _validate_params(self):
        super()._validate_params()
        if self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost classifier estimator does not support `qid_col` param."
            )
        if self.getOrDefault(self.objective):  # pylint: disable=no-member
            raise ValueError(
                "Setting custom 'objective' param is not allowed in 'SparkXGBClassifier'."
            )


_set_pyspark_xgb_cls_param_attrs(SparkXGBClassifier, SparkXGBClassifierModel)


class SparkXGBRanker(_SparkXGBEstimator):
    """SparkXGBRanker is a PySpark ML estimator. It implements the XGBoost
    ranking algorithm based on XGBoost python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBRanker automatically supports most of the parameters in
    `xgboost.XGBRanker` constructor and most of the parameters used in
    :py:class:`xgboost.XGBRanker` fit and predict method.

    SparkXGBRanker doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    SparkXGBRanker doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBRanker doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `raw_prediction_col` param doc below for more details.

    SparkXGBRanker doesn't support `validate_features` and `output_margin` param.

    SparkXGBRanker doesn't support setting `nthread` xgboost param, instead, the `nthread`
    param for each xgboost worker will be set equal to `spark.task.cpus` config value.


    Parameters
    ----------

    callbacks:
        The export and import of the callback functions are at best effort. For
        details, see :py:attr:`xgboost.spark.SparkXGBRanker.callbacks` param doc.
    validation_indicator_col:
        For params related to `xgboost.XGBRanker` training with
        evaluation dataset's supervision,
        set :py:attr:`xgboost.spark.XGBRanker.validation_indicator_col`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBRanker`
        fit method.
    weight_col:
        To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.SparkXGBRanker.weight_col` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBRanker`
        fit method.
    xgb_model:
        Set the value to be the instance returned by
        :func:`xgboost.spark.SparkXGBRankerModel.get_booster`.
    num_workers:
        Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one spark task.
    use_gpu:
        Boolean that specifies whether the executors are running on GPU
        instances.
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBRanker.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBRanker` fit method.
    qid_col:
        To specify the qid of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBRanker.qid_col` parameter
        instead of setting `qid` / `group`, `eval_qid` / `eval_group` in the
        `xgboost.XGBRanker` fit method.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    Examples
    --------

    >>> from xgboost.spark import SparkXGBRanker
    >>> from pyspark.ml.linalg import Vectors
    >>> ranker = SparkXGBRanker(qid_col="qid")
    >>> df_train = spark.createDataFrame(
    ...     [
    ...         (Vectors.dense(1.0, 2.0, 3.0), 0, 0),
    ...         (Vectors.dense(4.0, 5.0, 6.0), 1, 0),
    ...         (Vectors.dense(9.0, 4.0, 8.0), 2, 0),
    ...         (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 1),
    ...         (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 1),
    ...         (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 1),
    ...     ],
    ...     ["features", "label", "qid"],
    ... )
    >>> df_test = spark.createDataFrame(
    ...     [
    ...         (Vectors.dense(1.5, 2.0, 3.0), 0),
    ...         (Vectors.dense(4.5, 5.0, 6.0), 0),
    ...         (Vectors.dense(9.0, 4.5, 8.0), 0),
    ...         (Vectors.sparse(3, {1: 1.0, 2: 6.0}), 1),
    ...         (Vectors.sparse(3, {1: 6.0, 2: 7.0}), 1),
    ...         (Vectors.sparse(3, {1: 8.0, 2: 10.5}), 1),
    ...     ],
    ...     ["features", "qid"],
    ... )
    >>> model = ranker.fit(df_train)
    >>> model.transform(df_test).show()
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.setParams(**kwargs)

    @classmethod
    def _xgb_cls(cls):
        return XGBRanker

    @classmethod
    def _pyspark_model_cls(cls):
        return SparkXGBRankerModel

    def _validate_params(self):
        super()._validate_params()
        if not self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost ranker estimator requires setting `qid_col` param."
            )


_set_pyspark_xgb_cls_param_attrs(SparkXGBRanker, SparkXGBRankerModel)
