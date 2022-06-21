from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRegressor
from .core import (_XgboostEstimator, XgboostClassifierModel,
                   XgboostRegressorModel, _set_pyspark_xgb_cls_param_attrs)


class XgboostRegressor(_XgboostEstimator):
    """
    XgboostRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    XgboostRegressor automatically supports most of the parameters in
    `xgboost.XGBRegressor` constructor and most of the parameters used in
    `xgboost.XGBRegressor` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBRegressor>`_ for details).

    XgboostRegressor doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    XgboostRegressor doesn't support setting `base_margin` explicitly as well, but support
    another param called `baseMarginCol`. see doc below for more details.

    XgboostRegressor doesn't support `validate_features` and `output_margin` param.

    :param callbacks: The export and import of the callback functions are at best effort.
        For details, see :py:attr:`xgboost.spark.XgboostRegressor.callbacks` param doc.
    :param missing: The parameter `missing` in XgboostRegressor has different semantics with
        that in `xgboost.XGBRegressor`. For details, see
        :py:attr:`xgboost.spark.XgboostRegressor.missing` param doc.
    :param validationIndicatorCol: For params related to `xgboost.XGBRegressor` training
        with evaluation dataset's supervision, set
        :py:attr:`xgboost.spark.XgboostRegressor.validationIndicatorCol`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    :param weightCol: To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.XgboostRegressor.weightCol` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    :param xgb_model: Set the value to be the instance returned by
        :func:`xgboost.spark.XgboostRegressorModel.get_booster`.
    :param num_workers: Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one spark task.
    :param use_gpu: Boolean that specifies whether the executors are running on GPU
        instances.
    :param use_external_storage: Boolean that specifices whether you want to use
        external storage when training in a distributed manner. This allows using disk
        as cache. Setting this to true is useful when you want better memory utilization
        but is not needed for small test datasets.
    :param baseMarginCol: To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.XgboostRegressor.baseMarginCol` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBRegressor` fit method. Note: this isn't available for distributed
        training.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from xgboost.spark import XgboostRegressor
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
    >>> xgb_regressor = XgboostRegressor(max_depth=5, missing=0.0,
    ... validationIndicatorCol='isVal', weightCol='weight',
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
        return XgboostRegressorModel


_set_pyspark_xgb_cls_param_attrs(XgboostRegressor, XgboostRegressorModel)


class XgboostClassifier(_XgboostEstimator, HasProbabilityCol,
                        HasRawPredictionCol):
    """
    XgboostClassifier is a PySpark ML estimator. It implements the XGBoost classification
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    XgboostClassifier automatically supports most of the parameters in
    `xgboost.XGBClassifier` constructor and most of the parameters used in
    `xgboost.XGBClassifier` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBClassifier>`_ for details).

    XgboostClassifier doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    XgboostClassifier doesn't support setting `base_margin` explicitly as well, but support
    another param called `baseMarginCol`. see doc below for more details.

    XgboostClassifier doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `rawPredictionCol` param doc below for more details.

    XgboostClassifier doesn't support `validate_features` and `output_margin` param.

    :param callbacks: The export and import of the callback functions are at best effort. For
        details, see :py:attr:`xgboost.spark.XgboostClassifier.callbacks` param doc.
    :param missing: The parameter `missing` in XgboostClassifier has different semantics with
        that in `xgboost.XGBClassifier`. For details, see
        :py:attr:`xgboost.spark.XgboostClassifier.missing` param doc.
    :param rawPredictionCol: The `output_margin=True` is implicitly supported by the
        `rawPredictionCol` output column, which is always returned with the predicted margin
        values.
    :param validationIndicatorCol: For params related to `xgboost.XGBClassifier` training with
        evaluation dataset's supervision,
        set :py:attr:`xgboost.spark.XgboostClassifier.validationIndicatorCol`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    :param weightCol: To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.XgboostClassifier.weightCol` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    :param xgb_model: Set the value to be the instance returned by
        :func:`xgboost.spark.XgboostClassifierModel.get_booster`.
    :param num_workers: Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one spark task.
    :param use_gpu: Boolean that specifies whether the executors are running on GPU
        instances.
    :param use_external_storage: Boolean that specifices whether you want to use
        external storage when training in a distributed manner. This allows using disk
        as cache. Setting this to true is useful when you want better memory utilization
        but is not needed for small test datasets.
    :param baseMarginCol: To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.XgboostClassifier.baseMarginCol` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBClassifier` fit method. Note: this isn't available for distributed
        training.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from xgboost.spark import XgboostClassifier
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
    >>> xgb_classifier = XgboostClassifier(max_depth=5, missing=0.0,
    ...     validationIndicatorCol='isVal', weightCol='weight',
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
        return XgboostClassifierModel


_set_pyspark_xgb_cls_param_attrs(XgboostClassifier, XgboostClassifierModel)
