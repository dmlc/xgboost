"""Xgboost pyspark integration submodule for estimator API."""

# pylint: disable=protected-access, no-member, invalid-name
# pylint: disable=unused-argument, too-many-locals

from typing import Any, List, Optional, Type, Union

import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol

from ..collective import Config
from ..sklearn import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
    _ClassificationModel,
    _SparkXGBEstimator,
    _SparkXGBModel,
)
from .utils import get_class_name


def _set_pyspark_xgb_cls_param_attrs(
    estimator: Type[_SparkXGBEstimator], model: Type[_SparkXGBModel]
) -> None:
    """This function automatically infer to xgboost parameters and set them
    into corresponding pyspark estimators and models"""
    params_dict = estimator._get_xgb_params_default()

    def param_value_converter(v: Any) -> Any:
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        if isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        if isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        return v

    def set_param_attrs(attr_name: str, param: Param) -> None:
        param.typeConverter = param_value_converter
        setattr(estimator, attr_name, param)
        setattr(model, attr_name, param)

    for name in params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of "
            f"{get_class_name(estimator._xgb_cls())} for this param {name}"
        )

        param_obj: Param = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    fit_params_dict = estimator._get_fit_params_default()
    for name in fit_params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of {get_class_name(estimator._xgb_cls())}"
            f".fit() for this param {name}"
        )
        if name == "callbacks":
            doc += (
                "The callbacks can be arbitrary functions. It is saved using cloudpickle "
                "which is not a fully self-contained format. It may fail to load with "
                "different versions of dependencies."
            )
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    predict_params_dict = estimator._get_predict_params_default()
    for name in predict_params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of {get_class_name(estimator._xgb_cls())}"
            f".predict() for this param {name}"
        )
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)


class SparkXGBRegressor(_SparkXGBEstimator):
    """SparkXGBRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like
    - :py:class:`~pyspark.ml.tuning.CrossValidator`/
    - :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    - :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBRegressor automatically supports most of the parameters in
    :py:class:`xgboost.XGBRegressor` constructor and most of the parameters used in
    :py:meth:`xgboost.XGBRegressor.fit` and :py:meth:`xgboost.XGBRegressor.predict`
    method.

    To enable GPU support, set `device` to `cuda` or `gpu`.

    SparkXGBRegressor doesn't support setting `base_margin` explicitly as well, but
    support another param called `base_margin_col`. see doc below for more details.

    SparkXGBRegressor doesn't support `validate_features` and `output_margin` param.

    SparkXGBRegressor doesn't support setting `nthread` xgboost param, instead, the
    `nthread` param for each xgboost worker will be set equal to `spark.task.cpus`
    config value.


    Parameters
    ----------

    features_col:
        When the value is string, it requires the features column name to be vector type.
        When the value is a list of string, it requires all the feature columns to be numeric types.
    label_col:
        Label column name. Default to "label".
    prediction_col:
        Prediction column name. Default to "prediction"
    pred_contrib_col:
        Contribution prediction column name.
    validation_indicator_col:
        For params related to `xgboost.XGBRegressor` training with
        evaluation dataset's supervision,
        set :py:attr:`xgboost.spark.SparkXGBRegressor.validation_indicator_col`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    weight_col:
        To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.SparkXGBRegressor.weight_col` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBRegressor.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBRegressor` fit method.

    num_workers:
        How many XGBoost workers to be used to train.
        Each XGBoost worker corresponds to one spark task.
    device:

        .. versionadded:: 2.0.0

        Device for XGBoost workers, available options are `cpu`, `cuda`, and `gpu`.

    force_repartition:
        Boolean value to specify if forcing the input dataset to be repartitioned
        before XGBoost training.
    repartition_random_shuffle:
        Boolean value to specify if randomly shuffling the dataset when repartitioning is required.
    enable_sparse_data_optim:
        Boolean value to specify if enabling sparse data optimization, if True,
        Xgboost DMatrix object will be constructed from sparse matrix instead of
        dense matrix.
    launch_tracker_on_driver:
        Boolean value to indicate whether the tracker should be launched on the driver side or
        the executor side.
    coll_cfg:
        The collective configuration. See :py:class:`~xgboost.collective.Config`

    kwargs:
        A dictionary of xgboost parameters, please refer to
        https://xgboost.readthedocs.io/en/stable/parameter.html

    Note
    ----

    The Parameters chart above contains parameters that need special handling.
    For a full list of parameters, see entries with `Param(parent=...` below.

    This API is experimental.


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

    @keyword_only
    def __init__(  # pylint:disable=too-many-arguments
        self,
        *,
        features_col: Union[str, List[str]] = "features",
        label_col: str = "label",
        prediction_col: str = "prediction",
        pred_contrib_col: Optional[str] = None,
        validation_indicator_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        base_margin_col: Optional[str] = None,
        num_workers: int = 1,
        device: Optional[str] = None,
        force_repartition: bool = False,
        repartition_random_shuffle: bool = False,
        enable_sparse_data_optim: bool = False,
        launch_tracker_on_driver: bool = True,
        coll_cfg: Optional[Config] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        input_kwargs = self._input_kwargs
        self.setParams(**input_kwargs)

    @classmethod
    def _xgb_cls(cls) -> Type[XGBRegressor]:
        return XGBRegressor

    @classmethod
    def _pyspark_model_cls(cls) -> Type["SparkXGBRegressorModel"]:
        return SparkXGBRegressorModel

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost regressor estimator does not support `qid_col` param."
            )


class SparkXGBRegressorModel(_SparkXGBModel):
    """
    The model returned by :func:`xgboost.spark.SparkXGBRegressor.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls) -> Type[XGBRegressor]:
        return XGBRegressor


_set_pyspark_xgb_cls_param_attrs(SparkXGBRegressor, SparkXGBRegressorModel)


class SparkXGBClassifier(_SparkXGBEstimator, HasProbabilityCol, HasRawPredictionCol):
    """SparkXGBClassifier is a PySpark ML estimator. It implements the XGBoost
    classification algorithm based on XGBoost python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    - :py:class:`~pyspark.ml.tuning.CrossValidator`/
    - :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    - :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBClassifier automatically supports most of the parameters in
    :py:class:`xgboost.XGBClassifier` constructor and most of the parameters used in
    :py:meth:`xgboost.XGBClassifier.fit` and :py:meth:`xgboost.XGBClassifier.predict`
    method.

    To enable GPU support, set `device` to `cuda` or `gpu`.

    SparkXGBClassifier doesn't support setting `base_margin` explicitly as well, but
    support another param called `base_margin_col`. see doc below for more details.

    SparkXGBClassifier doesn't support setting `output_margin`, but we can get output
    margin from the raw prediction column. See `raw_prediction_col` param doc below for
    more details.

    SparkXGBClassifier doesn't support `validate_features` and `output_margin` param.

    SparkXGBClassifier doesn't support setting `nthread` xgboost param, instead, the
    `nthread` param for each xgboost worker will be set equal to `spark.task.cpus`
    config value.


    Parameters
    ----------

    features_col:
        When the value is string, it requires the features column name to be vector type.
        When the value is a list of string, it requires all the feature columns to be numeric types.
    label_col:
        Label column name. Default to "label".
    prediction_col:
        Prediction column name. Default to "prediction"
    probability_col:
        Column name for predicted class conditional probabilities. Default to probabilityCol
    raw_prediction_col:
        The `output_margin=True` is implicitly supported by the
        `rawPredictionCol` output column, which is always returned with the predicted margin
        values.
    pred_contrib_col:
        Contribution prediction column name.
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
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBClassifier.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBClassifier` fit method.

    num_workers:
        How many XGBoost workers to be used to train.
        Each XGBoost worker corresponds to one spark task.
    device:

        .. versionadded:: 2.0.0

        Device for XGBoost workers, available options are `cpu`, `cuda`, and `gpu`.

    force_repartition:
        Boolean value to specify if forcing the input dataset to be repartitioned
        before XGBoost training.
    repartition_random_shuffle:
        Boolean value to specify if randomly shuffling the dataset when repartitioning is required.
    enable_sparse_data_optim:
        Boolean value to specify if enabling sparse data optimization, if True,
        Xgboost DMatrix object will be constructed from sparse matrix instead of
        dense matrix.
    launch_tracker_on_driver:
        Boolean value to indicate whether the tracker should be launched on the driver side or
        the executor side.
    coll_cfg:
        The collective configuration. See :py:class:`~xgboost.collective.Config`

    kwargs:
        A dictionary of xgboost parameters, please refer to
        https://xgboost.readthedocs.io/en/stable/parameter.html

    Note
    ----

    The Parameters chart above contains parameters that need special handling.
    For a full list of parameters, see entries with `Param(parent=...` below.

    This API is experimental.

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

    @keyword_only
    def __init__(  # pylint:disable=too-many-arguments
        self,
        *,
        features_col: Union[str, List[str]] = "features",
        label_col: str = "label",
        prediction_col: str = "prediction",
        probability_col: str = "probability",
        raw_prediction_col: str = "rawPrediction",
        pred_contrib_col: Optional[str] = None,
        validation_indicator_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        base_margin_col: Optional[str] = None,
        num_workers: int = 1,
        device: Optional[str] = None,
        force_repartition: bool = False,
        repartition_random_shuffle: bool = False,
        enable_sparse_data_optim: bool = False,
        launch_tracker_on_driver: bool = True,
        coll_cfg: Optional[Config] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # The default 'objective' param value comes from sklearn `XGBClassifier` ctor,
        # but in pyspark we will automatically set objective param depending on
        # binary or multinomial input dataset, and we need to remove the fixed default
        # param value as well to avoid causing ambiguity.
        input_kwargs = self._input_kwargs
        self.setParams(**input_kwargs)
        self._setDefault(objective=None)

    @classmethod
    def _xgb_cls(cls) -> Type[XGBClassifier]:
        return XGBClassifier

    @classmethod
    def _pyspark_model_cls(cls) -> Type["SparkXGBClassifierModel"]:
        return SparkXGBClassifierModel

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost classifier estimator does not support `qid_col` param."
            )
        if self.getOrDefault("objective"):  # pylint: disable=no-member
            raise ValueError(
                "Setting custom 'objective' param is not allowed in 'SparkXGBClassifier'."
            )


class SparkXGBClassifierModel(_ClassificationModel):
    """
    The model returned by :func:`xgboost.spark.SparkXGBClassifier.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls) -> Type[XGBClassifier]:
        return XGBClassifier


_set_pyspark_xgb_cls_param_attrs(SparkXGBClassifier, SparkXGBClassifierModel)


class SparkXGBRanker(_SparkXGBEstimator):
    """SparkXGBRanker is a PySpark ML estimator. It implements the XGBoost
    ranking algorithm based on XGBoost python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    SparkXGBRanker automatically supports most of the parameters in
    :py:class:`xgboost.XGBRanker` constructor and most of the parameters used in
    :py:meth:`xgboost.XGBRanker.fit` and :py:meth:`xgboost.XGBRanker.predict` method.

    To enable GPU support, set `device` to `cuda` or `gpu`.

    SparkXGBRanker doesn't support setting `base_margin` explicitly as well, but support
    another param called `base_margin_col`. see doc below for more details.

    SparkXGBRanker doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `raw_prediction_col` param doc below for more
    details.

    SparkXGBRanker doesn't support `validate_features` and `output_margin` param.

    SparkXGBRanker doesn't support setting `nthread` xgboost param, instead, the
    `nthread` param for each xgboost worker will be set equal to `spark.task.cpus`
    config value.


    Parameters
    ----------

    features_col:
        When the value is string, it requires the features column name to be vector type.
        When the value is a list of string, it requires all the feature columns to be numeric types.
    label_col:
        Label column name. Default to "label".
    prediction_col:
        Prediction column name. Default to "prediction"
    pred_contrib_col:
        Contribution prediction column name.
    validation_indicator_col:
        For params related to `xgboost.XGBRanker` training with
        evaluation dataset's supervision,
        set :py:attr:`xgboost.spark.SparkXGBRanker.validation_indicator_col`
        parameter instead of setting the `eval_set` parameter in :py:class:`xgboost.XGBRanker`
        fit method.
    weight_col:
        To specify the weight of the training and validation dataset, set
        :py:attr:`xgboost.spark.SparkXGBRanker.weight_col` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in :py:class:`xgboost.XGBRanker`
        fit method.
    base_margin_col:
        To specify the base margins of the training and validation
        dataset, set :py:attr:`xgboost.spark.SparkXGBRanker.base_margin_col` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        :py:class:`xgboost.XGBRanker` fit method.
    qid_col:
        Query id column name.
    num_workers:
        How many XGBoost workers to be used to train.
        Each XGBoost worker corresponds to one spark task.
    device:

        .. versionadded:: 2.0.0

        Device for XGBoost workers, available options are `cpu`, `cuda`, and `gpu`.

    force_repartition:
        Boolean value to specify if forcing the input dataset to be repartitioned
        before XGBoost training.
    repartition_random_shuffle:
        Boolean value to specify if randomly shuffling the dataset when repartitioning is required.
    enable_sparse_data_optim:
        Boolean value to specify if enabling sparse data optimization, if True,
        Xgboost DMatrix object will be constructed from sparse matrix instead of
        dense matrix.
    launch_tracker_on_driver:
        Boolean value to indicate whether the tracker should be launched on the driver side or
        the executor side.
    coll_cfg:
        The collective configuration. See :py:class:`~xgboost.collective.Config`

    kwargs:
        A dictionary of xgboost parameters, please refer to
        https://xgboost.readthedocs.io/en/stable/parameter.html

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

    @keyword_only
    def __init__(  # pylint:disable=too-many-arguments
        self,
        *,
        features_col: Union[str, List[str]] = "features",
        label_col: str = "label",
        prediction_col: str = "prediction",
        pred_contrib_col: Optional[str] = None,
        validation_indicator_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        base_margin_col: Optional[str] = None,
        qid_col: Optional[str] = None,
        num_workers: int = 1,
        device: Optional[str] = None,
        force_repartition: bool = False,
        repartition_random_shuffle: bool = False,
        enable_sparse_data_optim: bool = False,
        launch_tracker_on_driver: bool = True,
        coll_cfg: Optional[Config] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        input_kwargs = self._input_kwargs
        self.setParams(**input_kwargs)

    @classmethod
    def _xgb_cls(cls) -> Type[XGBRanker]:
        return XGBRanker

    @classmethod
    def _pyspark_model_cls(cls) -> Type["SparkXGBRankerModel"]:
        return SparkXGBRankerModel

    def _validate_params(self) -> None:
        super()._validate_params()
        if not self.isDefined(self.qid_col):
            raise ValueError(
                "Spark Xgboost ranker estimator requires setting `qid_col` param."
            )


class SparkXGBRankerModel(_SparkXGBModel):
    """
    The model returned by :func:`xgboost.spark.SparkXGBRanker.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls) -> Type[XGBRanker]:
        return XGBRanker


_set_pyspark_xgb_cls_param_attrs(SparkXGBRanker, SparkXGBRankerModel)
