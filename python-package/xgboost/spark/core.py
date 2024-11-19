"""XGBoost pyspark integration submodule for core code."""

import base64

# pylint: disable=fixme, too-many-ancestors, protected-access, no-member, invalid-name
# pylint: disable=too-few-public-methods, too-many-lines, too-many-branches
import json
import logging
import os
from collections import namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pyspark import RDD, SparkConf, SparkContext, cloudpickle
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasPredictionCol,
    HasProbabilityCol,
    HasRawPredictionCol,
    HasValidationIndicatorCol,
    HasWeightCol,
)
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, countDistinct, pandas_udf, rand, struct
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    IntegralType,
    LongType,
    ShortType,
)
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module

import xgboost
from xgboost import XGBClassifier
from xgboost.compat import is_cudf_available, is_cupy_available
from xgboost.core import Booster, _check_distributed_params
from xgboost.sklearn import DEFAULT_N_ESTIMATORS, XGBModel, _can_use_qdm
from xgboost.training import train as worker_train

from .._typing import ArrayLike
from .data import (
    _read_csr_matrix_from_unwrapped_spark_vec,
    alias,
    create_dmatrix_from_partitions,
    pred_contribs,
    stack_series,
)
from .params import (
    HasArbitraryParamsDict,
    HasBaseMarginCol,
    HasContribPredictionCol,
    HasEnableSparseDataOptim,
    HasFeaturesCols,
    HasQueryIdCol,
)
from .utils import (
    CommunicatorContext,
    _get_default_params_from_func,
    _get_gpu_id,
    _get_max_num_concurrent_tasks,
    _get_rabit_args,
    _get_spark_session,
    _is_local,
    _is_standalone_or_localcluster,
    deserialize_booster,
    deserialize_xgb_model,
    get_class_name,
    get_logger,
    get_logger_level,
    serialize_booster,
    use_cuda,
)

# Put pyspark specific params here, they won't be passed to XGBoost.
# like `validationIndicatorCol`, `base_margin_col`
_pyspark_specific_params = [
    "featuresCol",
    "labelCol",
    "weightCol",
    "rawPredictionCol",
    "predictionCol",
    "probabilityCol",
    "validationIndicatorCol",
    "base_margin_col",
    "arbitrary_params_dict",
    "force_repartition",
    "num_workers",
    "feature_names",
    "features_cols",
    "enable_sparse_data_optim",
    "qid_col",
    "repartition_random_shuffle",
    "pred_contrib_col",
    "use_gpu",
]

_non_booster_params = ["missing", "n_estimators", "feature_types", "feature_weights"]

_pyspark_param_alias_map = {
    "features_col": "featuresCol",
    "label_col": "labelCol",
    "weight_col": "weightCol",
    "raw_prediction_col": "rawPredictionCol",
    "prediction_col": "predictionCol",
    "probability_col": "probabilityCol",
    "validation_indicator_col": "validationIndicatorCol",
}

_inverse_pyspark_param_alias_map = {v: k for k, v in _pyspark_param_alias_map.items()}

_unsupported_xgb_params = [
    "gpu_id",  # we have "device" pyspark param instead.
    "enable_categorical",  # Use feature_types param to specify categorical feature instead
    "n_jobs",  # Do not allow user to set it, will use `spark.task.cpus` value instead.
    "nthread",  # Ditto
]

_unsupported_fit_params = {
    "sample_weight",  # Supported by spark param weightCol
    "eval_set",  # Supported by spark param validation_indicator_col
    "sample_weight_eval_set",  # Supported by spark param weight_col + validation_indicator_col
    "base_margin",  # Supported by spark param base_margin_col
    "base_margin_eval_set",  # Supported by spark param base_margin_col + validation_indicator_col
    "group",  # Use spark param `qid_col` instead
    "qid",  # Use spark param `qid_col` instead
    "eval_group",  # Use spark param `qid_col` instead
    "eval_qid",  # Use spark param `qid_col` instead
}

_unsupported_train_params = {
    "evals",  # Supported by spark param validation_indicator_col
    "evals_result",  # Won't support yet+
}

_unsupported_predict_params = {
    # for classification, we can use rawPrediction as margin
    "output_margin",
    "validate_features",  # TODO
    "base_margin",  # Use pyspark base_margin_col param instead.
}

# TODO: supply hint message for all other unsupported params.
_unsupported_params_hint_message = {
    "enable_categorical": "`xgboost.spark` estimators do not have 'enable_categorical' param, "
    "but you can set `feature_types` param and mark categorical features with 'c' string."
}

# Global prediction names
Pred = namedtuple(
    "Pred", ("prediction", "raw_prediction", "probability", "pred_contrib")
)
pred = Pred("prediction", "rawPrediction", "probability", "predContrib")

_INIT_BOOSTER_SAVE_PATH = "init_booster.json"

_LOG_TAG = "XGBoost-PySpark"


class _SparkXGBParams(
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
    HasPredictionCol,
    HasValidationIndicatorCol,
    HasArbitraryParamsDict,
    HasBaseMarginCol,
    HasFeaturesCols,
    HasEnableSparseDataOptim,
    HasQueryIdCol,
    HasContribPredictionCol,
):
    num_workers = Param(
        Params._dummy(),
        "num_workers",
        "The number of XGBoost workers. Each XGBoost worker corresponds to one spark task.",
        TypeConverters.toInt,
    )
    device = Param(
        Params._dummy(),
        "device",
        (
            "The device type for XGBoost executors. Available options are `cpu`,`cuda`"
            " and `gpu`. Set `device` to `cuda` or `gpu` if the executors are running "
            "on GPU instances. Currently, only one GPU per task is supported."
        ),
        TypeConverters.toString,
    )
    use_gpu = Param(
        Params._dummy(),
        "use_gpu",
        (
            "Deprecated, use `device` instead. A boolean variable. Set use_gpu=true "
            "if the executors are running on GPU instances. Currently, only one GPU per"
            " task is supported."
        ),
        TypeConverters.toBoolean,
    )
    force_repartition = Param(
        Params._dummy(),
        "force_repartition",
        "A boolean variable. Set force_repartition=true if you "
        + "want to force the input dataset to be repartitioned before XGBoost training."
        + "Note: The auto repartitioning judgement is not fully accurate, so it is recommended"
        + "to have force_repartition be True.",
        TypeConverters.toBoolean,
    )
    repartition_random_shuffle = Param(
        Params._dummy(),
        "repartition_random_shuffle",
        "A boolean variable. Set repartition_random_shuffle=true if you want to random shuffle "
        "dataset when repartitioning is required. By default is True.",
        TypeConverters.toBoolean,
    )
    feature_names = Param(
        Params._dummy(),
        "feature_names",
        "A list of str to specify feature names.",
        TypeConverters.toList,
    )

    def set_device(self, value: str) -> "_SparkXGBParams":
        """Set device, optional value: cpu, cuda, gpu"""
        _check_distributed_params({"device": value})
        assert value in ("cpu", "cuda", "gpu")
        self.set(self.device, value)
        return self

    @classmethod
    def _xgb_cls(cls) -> Type[XGBModel]:
        """
        Subclasses should override this method and
        returns an xgboost.XGBModel subclass
        """
        raise NotImplementedError()

    # Parameters for xgboost.XGBModel()
    @classmethod
    def _get_xgb_params_default(cls) -> Dict[str, Any]:
        """Get the xgboost.sklearn.XGBModel default parameters and filter out some"""
        xgb_model_default = cls._xgb_cls()()
        params_dict = xgb_model_default.get_params()
        filtered_params_dict = {
            k: params_dict[k] for k in params_dict if k not in _unsupported_xgb_params
        }
        filtered_params_dict["n_estimators"] = DEFAULT_N_ESTIMATORS
        return filtered_params_dict

    def _set_xgb_params_default(self) -> None:
        """Set xgboost parameters into spark parameters"""
        filtered_params_dict = self._get_xgb_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_xgb_params_dict(
        self, gen_xgb_sklearn_estimator_param: bool = False
    ) -> Dict[str, Any]:
        """Generate the xgboost parameters which will be passed into xgboost library"""
        xgb_params = {}
        non_xgb_params = (
            set(_pyspark_specific_params)
            | self._get_fit_params_default().keys()
            | self._get_predict_params_default().keys()
        )
        if not gen_xgb_sklearn_estimator_param:
            non_xgb_params |= set(_non_booster_params)
        for param in self.extractParamMap():
            if param.name not in non_xgb_params:
                xgb_params[param.name] = self.getOrDefault(param)

        arbitrary_params_dict = self.getOrDefault(
            self.getParam("arbitrary_params_dict")
        )
        xgb_params.update(arbitrary_params_dict)
        return xgb_params

    # Parameters for xgboost.XGBModel().fit()
    @classmethod
    def _get_fit_params_default(cls) -> Dict[str, Any]:
        """Get the xgboost.XGBModel().fit() parameters"""
        fit_params = _get_default_params_from_func(
            cls._xgb_cls().fit, _unsupported_fit_params
        )
        return fit_params

    def _set_fit_params_default(self) -> None:
        """Get the xgboost.XGBModel().fit() parameters and set them to spark parameters"""
        filtered_params_dict = self._get_fit_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_fit_params_dict(self) -> Dict[str, Any]:
        """Generate the fit parameters which will be passed into fit function"""
        fit_params_keys = self._get_fit_params_default().keys()
        fit_params = {}
        for param in self.extractParamMap():
            if param.name in fit_params_keys:
                fit_params[param.name] = self.getOrDefault(param)
        return fit_params

    @classmethod
    def _get_predict_params_default(cls) -> Dict[str, Any]:
        """Get the parameters from xgboost.XGBModel().predict()"""
        predict_params = _get_default_params_from_func(
            cls._xgb_cls().predict, _unsupported_predict_params
        )
        return predict_params

    def _set_predict_params_default(self) -> None:
        """Get the parameters from xgboost.XGBModel().predict() and
        set them into spark parameters"""
        filtered_params_dict = self._get_predict_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_predict_params_dict(self) -> Dict[str, Any]:
        """Generate predict parameters which will be passed into xgboost.XGBModel().predict()"""
        predict_params_keys = self._get_predict_params_default().keys()
        predict_params = {}
        for param in self.extractParamMap():
            if param.name in predict_params_keys:
                predict_params[param.name] = self.getOrDefault(param)
        return predict_params

    def _validate_gpu_params(
        self, spark_version: str, conf: SparkConf, is_local: bool = False
    ) -> None:
        """Validate the gpu parameters and gpu configurations"""

        if self._run_on_gpu():
            if is_local:
                # Supporting GPU training in Spark local mode is just for debugging
                # purposes, so it's okay for printing the below warning instead of
                # checking the real gpu numbers and raising the exception.
                get_logger(self.__class__.__name__).warning(
                    "You have enabled GPU in spark local mode. Please make sure your"
                    " local node has at least %d GPUs",
                    self.getOrDefault(self.num_workers),
                )
            else:
                executor_gpus = conf.get("spark.executor.resource.gpu.amount")
                if executor_gpus is None:
                    raise ValueError(
                        "The `spark.executor.resource.gpu.amount` is required for training"
                        " on GPU."
                    )
                gpu_per_task = conf.get("spark.task.resource.gpu.amount")
                if gpu_per_task is not None and float(gpu_per_task) > 1.0:
                    get_logger(self.__class__.__name__).warning(
                        "The configuration assigns %s GPUs to each Spark task, but each "
                        "XGBoost training task only utilizes 1 GPU, which will lead to "
                        "unnecessary GPU waste",
                        gpu_per_task,
                    )
                # For 3.5.1+, Spark supports task stage-level scheduling for
                #                          Yarn/K8s/Standalone/Local cluster
                # From 3.4.0 ~ 3.5.0, Spark only supports task stage-level scheduing for
                #                           Standalone/Local cluster
                # For spark below 3.4.0, Task stage-level scheduling is not supported.
                #
                # With stage-level scheduling, spark.task.resource.gpu.amount is not required
                # to be set explicitly. Or else, spark.task.resource.gpu.amount is a must-have and
                # must be set to 1.0
                if spark_version < "3.4.0" or (
                    "3.4.0" <= spark_version < "3.5.1"
                    and not _is_standalone_or_localcluster(conf)
                ):
                    if gpu_per_task is not None:
                        if float(gpu_per_task) < 1.0:
                            raise ValueError(
                                "XGBoost doesn't support GPU fractional configurations. Please set "
                                "`spark.task.resource.gpu.amount=spark.executor.resource.gpu."
                                "amount`. To enable GPU fractional configurations, you can try "
                                "standalone/localcluster with spark 3.4.0+ and"
                                "YARN/K8S with spark 3.5.1+"
                            )
                    else:
                        raise ValueError(
                            "The `spark.task.resource.gpu.amount` is required for training"
                            " on GPU."
                        )

    def _validate_params(self) -> None:
        # pylint: disable=too-many-branches
        init_model = self.getOrDefault("xgb_model")
        if init_model is not None and not isinstance(init_model, Booster):
            raise ValueError(
                "The xgb_model param must be set with a `xgboost.core.Booster` "
                "instance."
            )

        if self.getOrDefault(self.num_workers) < 1:
            raise ValueError(
                f"Number of workers was {self.getOrDefault(self.num_workers)}."
                f"It cannot be less than 1 [Default is 1]"
            )

        tree_method = self.getOrDefault(self.getParam("tree_method"))
        if tree_method == "exact":
            raise ValueError(
                "The `exact` tree method is not supported for distributed systems."
            )

        if self.getOrDefault(self.features_cols):
            if not self._run_on_gpu():
                raise ValueError(
                    "features_col param with list value requires `device=cuda`."
                )

        if self.getOrDefault("objective") is not None:
            if not isinstance(self.getOrDefault("objective"), str):
                raise ValueError("Only string type 'objective' param is allowed.")

        eval_metric = "eval_metric"
        if self.getOrDefault(eval_metric) is not None:
            if not (
                isinstance(self.getOrDefault(eval_metric), str)
                or (
                    isinstance(self.getOrDefault(eval_metric), List)
                    and all(
                        isinstance(metric, str)
                        for metric in self.getOrDefault(eval_metric)
                    )
                )
            ):
                raise ValueError(
                    "Only string type or list of string type 'eval_metric' param is allowed."
                )

        if self.getOrDefault("early_stopping_rounds") is not None:
            if not (
                self.isDefined(self.validationIndicatorCol)
                and self.getOrDefault(self.validationIndicatorCol) != ""
            ):
                raise ValueError(
                    "If 'early_stopping_rounds' param is set, you need to set "
                    "'validation_indicator_col' param as well."
                )

        if self.getOrDefault(self.enable_sparse_data_optim):
            if self.getOrDefault("missing") != 0.0:
                # If DMatrix is constructed from csr / csc matrix, then inactive elements
                # in csr / csc matrix are regarded as missing value, but, in pyspark, we
                # are hard to control elements to be active or inactive in sparse vector column,
                # some spark transformers such as VectorAssembler might compress vectors
                # to be dense or sparse format automatically, and when a spark ML vector object
                # is compressed to sparse vector, then all zero value elements become inactive.
                # So we force setting missing param to be 0 when enable_sparse_data_optim config
                # is True.
                raise ValueError(
                    "If enable_sparse_data_optim is True, missing param != 0 is not supported."
                )
            if self.getOrDefault(self.features_cols):
                raise ValueError(
                    "If enable_sparse_data_optim is True, you cannot set multiple feature columns "
                    "but you should set one feature column with values of "
                    "`pyspark.ml.linalg.Vector` type."
                )

        ss = _get_spark_session()
        sc = ss.sparkContext
        self._validate_gpu_params(ss.version, sc.getConf(), _is_local(sc))

    def _run_on_gpu(self) -> bool:
        """If train or transform on the gpu according to the parameters"""

        return (
            use_cuda(self.getOrDefault(self.device))
            or self.getOrDefault(self.use_gpu)
            or self.getOrDefault(self.getParam("tree_method")) == "gpu_hist"
        )


def _validate_and_convert_feature_col_as_float_col_list(
    dataset: DataFrame, features_col_names: List[str]
) -> List[Column]:
    """Values in feature columns must be integral types or float/double types"""
    feature_cols = []
    for c in features_col_names:
        if isinstance(dataset.schema[c].dataType, DoubleType):
            feature_cols.append(col(c).cast(FloatType()).alias(c))
        elif isinstance(dataset.schema[c].dataType, (FloatType, IntegralType)):
            feature_cols.append(col(c))
        else:
            raise ValueError(
                "Values in feature columns must be integral types or float/double types."
            )
    return feature_cols


def _validate_and_convert_feature_col_as_array_col(
    dataset: DataFrame, features_col_name: str
) -> Column:
    """It handles
    1. Convert vector type to array type
    2. Cast to Array(Float32)"""
    features_col_datatype = dataset.schema[features_col_name].dataType
    features_col = col(features_col_name)
    if isinstance(features_col_datatype, ArrayType):
        if not isinstance(
            features_col_datatype.elementType,
            (DoubleType, FloatType, LongType, IntegerType, ShortType),
        ):
            raise ValueError(
                "If feature column is array type, its elements must be number type, "
                f"got {features_col_datatype.elementType}."
            )
        features_array_col = features_col.cast(ArrayType(FloatType())).alias(alias.data)
    elif isinstance(features_col_datatype, VectorUDT):
        features_array_col = vector_to_array(features_col, dtype="float32").alias(
            alias.data
        )
    else:
        raise ValueError(
            "feature column must be array type or `pyspark.ml.linalg.Vector` type, "
            "if you want to use multiple numetric columns as features, please use "
            "`pyspark.ml.transform.VectorAssembler` to assemble them into a vector "
            "type column first."
        )
    return features_array_col


def _get_unwrap_udt_fn() -> Callable[[Union[Column, str]], Column]:
    try:
        from pyspark.sql.functions import unwrap_udt

        return unwrap_udt
    except ImportError:
        pass

    try:
        from pyspark.databricks.sql.functions import unwrap_udt as databricks_unwrap_udt

        return databricks_unwrap_udt
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import pyspark `unwrap_udt` function. Please install pyspark>=3.4 "
            "or run on Databricks Runtime."
        ) from exc


def _get_unwrapped_vec_cols(feature_col: Column) -> List[Column]:
    unwrap_udt = _get_unwrap_udt_fn()
    features_unwrapped_vec_col = unwrap_udt(feature_col)

    # After a `pyspark.ml.linalg.VectorUDT` type column being unwrapped, it becomes
    # a pyspark struct type column, the struct fields are:
    #  - `type`: byte
    #  - `size`: int
    #  - `indices`: array<int>
    #  - `values`: array<double>
    # For sparse vector, `type` field is 0, `size` field means vector length,
    # `indices` field is the array of active element indices, `values` field
    # is the array of active element values.
    # For dense vector, `type` field is 1, `size` and `indices` fields are None,
    # `values` field is the array of the vector element values.
    return [
        features_unwrapped_vec_col.type.alias("featureVectorType"),
        features_unwrapped_vec_col.size.alias("featureVectorSize"),
        features_unwrapped_vec_col.indices.alias("featureVectorIndices"),
        # Note: the value field is double array type, cast it to float32 array type
        # for speedup following repartitioning.
        features_unwrapped_vec_col.values.cast(ArrayType(FloatType())).alias(
            "featureVectorValues"
        ),
    ]


FeatureProp = namedtuple(
    "FeatureProp",
    ("enable_sparse_data_optim", "has_validation_col", "features_cols_names"),
)


_MODEL_CHUNK_SIZE = 4096 * 1024


class _SparkXGBEstimator(Estimator, _SparkXGBParams, MLReadable, MLWritable):
    _input_kwargs: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self._set_xgb_params_default()
        self._set_fit_params_default()
        self._set_predict_params_default()
        # Note: The default value for arbitrary_params_dict must always be empty dict.
        #  For additional settings added into "arbitrary_params_dict" by default,
        #  they are added in `setParams`.
        self._setDefault(
            num_workers=1,
            device="cpu",
            use_gpu=False,
            force_repartition=False,
            repartition_random_shuffle=False,
            feature_names=None,
            feature_types=None,
            arbitrary_params_dict={},
        )

        self.logger = get_logger(self.__class__.__name__)

    def setParams(self, **kwargs: Any) -> None:  # pylint: disable=invalid-name
        """
        Set params for the estimator.
        """
        _extra_params = {}
        if "arbitrary_params_dict" in kwargs:
            raise ValueError("Invalid param name: 'arbitrary_params_dict'.")

        for k, v in kwargs.items():
            # We're not allowing user use features_cols directly.
            if k == self.features_cols.name:
                raise ValueError(
                    f"Unsupported param '{k}' please use features_col instead."
                )
            if k in _inverse_pyspark_param_alias_map:
                raise ValueError(
                    f"Please use param name {_inverse_pyspark_param_alias_map[k]} instead."
                )
            if k in _pyspark_param_alias_map:
                if k == _inverse_pyspark_param_alias_map[
                    self.featuresCol.name
                ] and isinstance(v, list):
                    real_k = self.features_cols.name
                    k = real_k
                else:
                    real_k = _pyspark_param_alias_map[k]
                    k = real_k

            if self.hasParam(k):
                if k == "features_col" and isinstance(v, list):
                    self._set(**{"features_cols": v})
                else:
                    self._set(**{str(k): v})
            else:
                if (
                    k in _unsupported_xgb_params
                    or k in _unsupported_fit_params
                    or k in _unsupported_predict_params
                    or k in _unsupported_train_params
                ):
                    err_msg = _unsupported_params_hint_message.get(
                        k, f"Unsupported param '{k}'."
                    )
                    raise ValueError(err_msg)
                _extra_params[k] = v

        _check_distributed_params(kwargs)
        _existing_extra_params = self.getOrDefault(self.arbitrary_params_dict)
        self._set(arbitrary_params_dict={**_existing_extra_params, **_extra_params})

    @classmethod
    def _pyspark_model_cls(cls) -> Type["_SparkXGBModel"]:
        """
        Subclasses should override this method and
        returns a _SparkXGBModel subclass
        """
        raise NotImplementedError()

    def _create_pyspark_model(self, xgb_model: XGBModel) -> "_SparkXGBModel":
        return self._pyspark_model_cls()(xgb_model)

    def _convert_to_sklearn_model(self, booster: bytearray, config: str) -> XGBModel:
        xgb_sklearn_params = self._gen_xgb_params_dict(
            gen_xgb_sklearn_estimator_param=True
        )
        sklearn_model = self._xgb_cls()(**xgb_sklearn_params)
        sklearn_model.load_model(booster)
        sklearn_model._Booster.load_config(config)
        return sklearn_model

    def _repartition_needed(self, dataset: DataFrame) -> bool:
        """
        We repartition the dataset if the number of workers is not equal to the number of
        partitions."""
        if self.getOrDefault(self.force_repartition):
            return True
        num_workers = self.getOrDefault(self.num_workers)
        num_partitions = dataset.rdd.getNumPartitions()
        return not num_workers == num_partitions

    def _get_distributed_train_params(self, dataset: DataFrame) -> Dict[str, Any]:
        """
        This just gets the configuration params for distributed xgboost
        """
        params = self._gen_xgb_params_dict()
        fit_params = self._gen_fit_params_dict()
        verbose_eval = fit_params.pop("verbose", None)

        params.update(fit_params)
        params["verbose_eval"] = verbose_eval
        classification = self._xgb_cls() == XGBClassifier
        if classification:
            num_classes = int(
                dataset.select(countDistinct(alias.label)).collect()[0][0]
            )
            if num_classes <= 2:
                params["objective"] = "binary:logistic"
            else:
                params["objective"] = "multi:softprob"
                params["num_class"] = num_classes
        else:
            # use user specified objective or default objective.
            # e.g., the default objective for Regressor is 'reg:squarederror'
            params["objective"] = self.getOrDefault("objective")

        # TODO: support "num_parallel_tree" for random forest
        params["num_boost_round"] = self.getOrDefault("n_estimators")

        return params

    @classmethod
    def _get_xgb_train_call_args(
        cls, train_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        xgb_train_default_args = _get_default_params_from_func(
            xgboost.train, _unsupported_train_params
        )
        booster_params, kwargs_params = {}, {}
        for key, value in train_params.items():
            if key in xgb_train_default_args:
                kwargs_params[key] = value
            else:
                booster_params[key] = value

        booster_params = {
            k: v for k, v in booster_params.items() if k not in _non_booster_params
        }
        return booster_params, kwargs_params

    def _prepare_input_columns_and_feature_prop(
        self, dataset: DataFrame
    ) -> Tuple[List[Column], FeatureProp]:
        label_col = col(self.getOrDefault(self.labelCol)).alias(alias.label)

        select_cols = [label_col]
        features_cols_names = None
        enable_sparse_data_optim = self.getOrDefault(self.enable_sparse_data_optim)
        if enable_sparse_data_optim:
            features_col_name = self.getOrDefault(self.featuresCol)
            features_col_datatype = dataset.schema[features_col_name].dataType
            if not isinstance(features_col_datatype, VectorUDT):
                raise ValueError(
                    "If enable_sparse_data_optim is True, the feature column values must be "
                    "`pyspark.ml.linalg.Vector` type."
                )
            select_cols.extend(_get_unwrapped_vec_cols(col(features_col_name)))
        else:
            if self.getOrDefault(self.features_cols):
                features_cols_names = self.getOrDefault(self.features_cols)
                features_cols = _validate_and_convert_feature_col_as_float_col_list(
                    dataset, features_cols_names
                )
                select_cols.extend(features_cols)
            else:
                features_array_col = _validate_and_convert_feature_col_as_array_col(
                    dataset, self.getOrDefault(self.featuresCol)
                )
                select_cols.append(features_array_col)

        if self.isDefined(self.weightCol) and self.getOrDefault(self.weightCol) != "":
            select_cols.append(
                col(self.getOrDefault(self.weightCol)).alias(alias.weight)
            )

        has_validation_col = False
        if (
            self.isDefined(self.validationIndicatorCol)
            and self.getOrDefault(self.validationIndicatorCol) != ""
        ):
            select_cols.append(
                col(self.getOrDefault(self.validationIndicatorCol)).alias(alias.valid)
            )
            # In some cases, see https://issues.apache.org/jira/browse/SPARK-40407,
            # the df.repartition can result in some reducer partitions without data,
            # which will cause exception or hanging issue when creating DMatrix.
            has_validation_col = True

        if (
            self.isDefined(self.base_margin_col)
            and self.getOrDefault(self.base_margin_col) != ""
        ):
            select_cols.append(
                col(self.getOrDefault(self.base_margin_col)).alias(alias.margin)
            )

        if self.isDefined(self.qid_col) and self.getOrDefault(self.qid_col) != "":
            select_cols.append(col(self.getOrDefault(self.qid_col)).alias(alias.qid))

        feature_prop = FeatureProp(
            enable_sparse_data_optim, has_validation_col, features_cols_names
        )
        return select_cols, feature_prop

    def _prepare_input(self, dataset: DataFrame) -> Tuple[DataFrame, FeatureProp]:
        """Prepare the input including column pruning, repartition and so on"""

        select_cols, feature_prop = self._prepare_input_columns_and_feature_prop(
            dataset
        )

        dataset = dataset.select(*select_cols)

        num_workers = self.getOrDefault(self.num_workers)
        sc = _get_spark_session().sparkContext
        max_concurrent_tasks = _get_max_num_concurrent_tasks(sc)

        if num_workers > max_concurrent_tasks:
            get_logger(self.__class__.__name__).warning(
                "The num_workers %s set for xgboost distributed "
                "training is greater than current max number of concurrent "
                "spark task slots, you need wait until more task slots available "
                "or you need increase spark cluster workers.",
                num_workers,
            )

        if self._repartition_needed(dataset):
            # If validationIndicatorCol defined, and if user unionise train and validation
            # dataset, users must set force_repartition to true to force repartition.
            # Or else some partitions might contain only train or validation dataset.
            if self.getOrDefault(self.repartition_random_shuffle):
                # In some cases, spark round-robin repartition might cause data skew
                # use random shuffle can address it.
                dataset = dataset.repartition(num_workers, rand(1))
            else:
                dataset = dataset.repartition(num_workers)

        if self.isDefined(self.qid_col) and self.getOrDefault(self.qid_col) != "":
            # XGBoost requires qid to be sorted for each partition
            dataset = dataset.sortWithinPartitions(alias.qid, ascending=True)

        return dataset, feature_prop

    def _get_xgb_parameters(
        self, dataset: DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        train_params = self._get_distributed_train_params(dataset)
        booster_params, train_call_kwargs_params = self._get_xgb_train_call_args(
            train_params
        )
        cpu_per_task = int(
            _get_spark_session().sparkContext.getConf().get("spark.task.cpus", "1")
        )

        dmatrix_kwargs = {
            "nthread": cpu_per_task,
            "feature_types": self.getOrDefault("feature_types"),
            "feature_names": self.getOrDefault("feature_names"),
            "feature_weights": self.getOrDefault("feature_weights"),
            "missing": float(self.getOrDefault("missing")),
        }
        if dmatrix_kwargs["feature_types"] is not None:
            dmatrix_kwargs["enable_categorical"] = True
        booster_params["nthread"] = cpu_per_task

        # Remove the parameters whose value is None
        booster_params = {k: v for k, v in booster_params.items() if v is not None}
        train_call_kwargs_params = {
            k: v for k, v in train_call_kwargs_params.items() if v is not None
        }
        dmatrix_kwargs = {k: v for k, v in dmatrix_kwargs.items() if v is not None}

        return booster_params, train_call_kwargs_params, dmatrix_kwargs

    def _skip_stage_level_scheduling(self, spark_version: str, conf: SparkConf) -> bool:
        # pylint: disable=too-many-return-statements
        """Check if stage-level scheduling is not needed,
        return true to skip stage-level scheduling"""

        if self._run_on_gpu():
            if spark_version < "3.4.0":
                self.logger.info(
                    "Stage-level scheduling in xgboost requires spark version 3.4.0+"
                )
                return True

            if (
                "3.4.0" <= spark_version < "3.5.1"
                and not _is_standalone_or_localcluster(conf)
            ):
                self.logger.info(
                    "For %s, Stage-level scheduling in xgboost requires spark standalone "
                    "or local-cluster mode",
                    spark_version,
                )
                return True

            executor_cores = conf.get("spark.executor.cores")
            executor_gpus = conf.get("spark.executor.resource.gpu.amount")
            if executor_cores is None or executor_gpus is None:
                self.logger.info(
                    "Stage-level scheduling in xgboost requires spark.executor.cores, "
                    "spark.executor.resource.gpu.amount to be set."
                )
                return True

            if int(executor_cores) == 1:
                # there will be only 1 task running at any time.
                self.logger.info(
                    "Stage-level scheduling in xgboost requires spark.executor.cores > 1 "
                )
                return True

            if int(executor_gpus) > 1:
                # For spark.executor.resource.gpu.amount > 1, we suppose user knows how to configure
                # to make xgboost run successfully.
                #
                self.logger.info(
                    "Stage-level scheduling in xgboost will not work "
                    "when spark.executor.resource.gpu.amount>1"
                )
                return True

            task_gpu_amount = conf.get("spark.task.resource.gpu.amount")

            if task_gpu_amount is None:
                # The ETL tasks will not grab a gpu when spark.task.resource.gpu.amount is not set,
                # but with stage-level scheduling, we can make training task grab the gpu.
                return False

            if float(task_gpu_amount) == float(executor_gpus):
                # spark.executor.resource.gpu.amount=spark.task.resource.gpu.amount "
                # results in only 1 task running at a time, which may cause perf issue.
                return True

            # We can enable stage-level scheduling
            return False

        # CPU training doesn't require stage-level scheduling
        return True

    def _try_stage_level_scheduling(self, rdd: RDD) -> RDD:
        """Try to enable stage-level scheduling"""
        ss = _get_spark_session()
        conf = ss.sparkContext.getConf()
        if _is_local(ss.sparkContext) or self._skip_stage_level_scheduling(
            ss.version, conf
        ):
            return rdd

        # executor_cores will not be None
        executor_cores = conf.get("spark.executor.cores")
        assert executor_cores is not None

        # Spark-rapids is a project to leverage GPUs to accelerate spark SQL.
        # If spark-rapids is enabled, to avoid GPU OOM, we don't allow other
        # ETL gpu tasks running alongside training tasks.
        spark_plugins = ss.conf.get("spark.plugins", " ")
        assert spark_plugins is not None
        spark_rapids_sql_enabled = ss.conf.get("spark.rapids.sql.enabled", "true")
        assert spark_rapids_sql_enabled is not None

        task_cores = (
            int(executor_cores)
            if "com.nvidia.spark.SQLPlugin" in spark_plugins
            and "true" == spark_rapids_sql_enabled.lower()
            else (int(executor_cores) // 2) + 1
        )

        # Each training task requires cpu cores > total executor cores//2 + 1 which can
        # make sure the tasks be sent to different executors.
        #
        # Please note that we can't use GPU to limit the concurrent tasks because of
        # https://issues.apache.org/jira/browse/SPARK-45527.

        task_gpus = 1.0
        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build

        self.logger.info(
            "XGBoost training tasks require the resource(cores=%s, gpu=%s).",
            task_cores,
            task_gpus,
        )
        return rdd.withResources(rp)

    def _fit(self, dataset: DataFrame) -> "_SparkXGBModel":
        # pylint: disable=too-many-statements, too-many-locals
        self._validate_params()

        dataset, feature_prop = self._prepare_input(dataset)

        (
            booster_params,
            train_call_kwargs_params,
            dmatrix_kwargs,
        ) = self._get_xgb_parameters(dataset)

        run_on_gpu = self._run_on_gpu()

        is_local = _is_local(_get_spark_session().sparkContext)

        num_workers = self.getOrDefault(self.num_workers)

        log_level = get_logger_level(_LOG_TAG)

        def _train_booster(
            pandas_df_iter: Iterator[pd.DataFrame],
        ) -> Iterator[pd.DataFrame]:
            """Takes in an RDD partition and outputs a booster for that partition after
            going through the Rabit Ring protocol

            """
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()

            dev_ordinal = None
            use_qdm = _can_use_qdm(booster_params.get("tree_method", None))
            verbosity = booster_params.get("verbosity", 1)
            msg = "Training on CPUs"
            if run_on_gpu:
                dev_ordinal = (
                    context.partitionId() if is_local else _get_gpu_id(context)
                )
                booster_params["device"] = "cuda:" + str(dev_ordinal)
                # If cuDF is not installed, then using DMatrix instead of QDM,
                # because without cuDF, DMatrix performs better than QDM.
                # Note: Checking `is_cudf_available` in spark worker side because
                # spark worker might has different python environment with driver side.
                use_qdm = use_qdm and is_cudf_available()
                msg = (
                    f"Leveraging {booster_params['device']} to train with "
                    f"QDM: {'on' if use_qdm else 'off'}"
                )

            if use_qdm and (booster_params.get("max_bin", None) is not None):
                dmatrix_kwargs["max_bin"] = booster_params["max_bin"]

            _rabit_args = {}
            if context.partitionId() == 0:
                _rabit_args = _get_rabit_args(context, num_workers)
                get_logger(_LOG_TAG, log_level).info(msg)

            worker_message = {
                "rabit_msg": _rabit_args,
                "use_qdm": use_qdm,
            }

            messages = context.allGather(message=json.dumps(worker_message))
            if len(set(json.loads(x)["use_qdm"] for x in messages)) != 1:
                raise RuntimeError("The workers' cudf environments are in-consistent ")

            _rabit_args = json.loads(messages[0])["rabit_msg"]

            evals_result: Dict[str, Any] = {}
            with CommunicatorContext(context, **_rabit_args):
                with xgboost.config_context(verbosity=verbosity):
                    dtrain, dvalid = create_dmatrix_from_partitions(
                        pandas_df_iter,
                        feature_prop.features_cols_names,
                        dev_ordinal,
                        use_qdm,
                        dmatrix_kwargs,
                        enable_sparse_data_optim=feature_prop.enable_sparse_data_optim,
                        has_validation_col=feature_prop.has_validation_col,
                    )
                if dvalid is not None:
                    dval = [(dtrain, "training"), (dvalid, "validation")]
                else:
                    dval = None
                booster = worker_train(
                    params=booster_params,
                    dtrain=dtrain,
                    evals=dval,
                    evals_result=evals_result,
                    **train_call_kwargs_params,
                )
            context.barrier()

            if context.partitionId() == 0:
                config = booster.save_config()
                yield pd.DataFrame({"data": [config]})
                booster_json = booster.save_raw("json").decode("utf-8")

                for offset in range(0, len(booster_json), _MODEL_CHUNK_SIZE):
                    booster_chunk = booster_json[offset : offset + _MODEL_CHUNK_SIZE]
                    yield pd.DataFrame({"data": [booster_chunk]})

        def _run_job() -> Tuple[str, str]:
            rdd = (
                dataset.mapInPandas(
                    _train_booster,  # type: ignore
                    schema="data string",
                )
                .rdd.barrier()
                .mapPartitions(lambda x: x)
            )
            rdd_with_resource = self._try_stage_level_scheduling(rdd)
            ret = rdd_with_resource.collect()
            data = [v[0] for v in ret]
            return data[0], "".join(data[1:])

        get_logger(_LOG_TAG).info(
            "Running xgboost-%s on %s workers with"
            "\n\tbooster params: %s"
            "\n\ttrain_call_kwargs_params: %s"
            "\n\tdmatrix_kwargs: %s",
            xgboost._py_version(),
            num_workers,
            booster_params,
            train_call_kwargs_params,
            dmatrix_kwargs,
        )
        (config, booster) = _run_job()
        get_logger(_LOG_TAG).info("Finished xgboost training!")

        result_xgb_model = self._convert_to_sklearn_model(
            bytearray(booster, "utf-8"), config
        )
        spark_model = self._create_pyspark_model(result_xgb_model)
        # According to pyspark ML convention, the model uid should be the same
        # with estimator uid.
        spark_model._resetUid(self.uid)
        return self._copyValues(spark_model)

    def write(self) -> "SparkXGBWriter":
        """
        Return the writer for saving the estimator.
        """
        return SparkXGBWriter(self)

    @classmethod
    def read(cls) -> "SparkXGBReader":
        """
        Return the reader for loading the estimator.
        """
        return SparkXGBReader(cls)


class _SparkXGBModel(Model, _SparkXGBParams, MLReadable, MLWritable):
    def __init__(self, xgb_sklearn_model: Optional[XGBModel] = None) -> None:
        super().__init__()
        self._xgb_sklearn_model = xgb_sklearn_model

    @classmethod
    def _xgb_cls(cls) -> Type[XGBModel]:
        raise NotImplementedError()

    def get_booster(self) -> Booster:
        """
        Return the `xgboost.core.Booster` instance.
        """
        assert self._xgb_sklearn_model is not None
        return self._xgb_sklearn_model.get_booster()

    def get_feature_importances(
        self, importance_type: str = "weight"
    ) -> Dict[str, Union[float, List[float]]]:
        """Get feature importance of each feature.
        Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        Parameters
        ----------
        importance_type: str, default 'weight'
            One of the importance types defined above.
        """
        return self.get_booster().get_score(importance_type=importance_type)

    def write(self) -> "SparkXGBModelWriter":
        """
        Return the writer for saving the model.
        """
        return SparkXGBModelWriter(self)

    @classmethod
    def read(cls) -> "SparkXGBModelReader":
        """
        Return the reader for loading the model.
        """
        return SparkXGBModelReader(cls)

    def _get_feature_col(
        self, dataset: DataFrame
    ) -> Tuple[List[Column], Optional[List[str]]]:
        """XGBoost model trained with features_cols parameter can also predict
        vector or array feature type. But first we need to check features_cols
        and then featuresCol
        """
        if self.getOrDefault(self.enable_sparse_data_optim):
            feature_col_names = None
            features_col = _get_unwrapped_vec_cols(
                col(self.getOrDefault(self.featuresCol))
            )
            return features_col, feature_col_names

        feature_col_names = self.getOrDefault(self.features_cols)
        features_col = []
        if feature_col_names and set(feature_col_names).issubset(set(dataset.columns)):
            # The model is trained with features_cols and the predicted dataset
            # also contains all the columns specified by features_cols.
            features_col = _validate_and_convert_feature_col_as_float_col_list(
                dataset, feature_col_names
            )
        else:
            # 1. The model was trained by features_cols, but the dataset doesn't contain
            #       all the columns specified by features_cols, so we need to check if
            #       the dataframe has the featuresCol
            # 2. The model was trained by featuresCol, and the predicted dataset must contain
            #       featuresCol column.
            feature_col_names = None
            features_col.append(
                _validate_and_convert_feature_col_as_array_col(
                    dataset, self.getOrDefault(self.featuresCol)
                )
            )
        return features_col, feature_col_names

    def _get_pred_contrib_col_name(self) -> Optional[str]:
        """Return the pred_contrib_col col name"""
        pred_contrib_col_name = None
        if (
            self.isDefined(self.pred_contrib_col)
            and self.getOrDefault(self.pred_contrib_col) != ""
        ):
            pred_contrib_col_name = self.getOrDefault(self.pred_contrib_col)

        return pred_contrib_col_name

    def _out_schema(self) -> Tuple[bool, str]:
        """Return the bool to indicate if it's a single prediction, true is single prediction,
        and the returned type of the user-defined function. The value must
        be a DDL-formatted type string."""

        if self._get_pred_contrib_col_name() is not None:
            return False, f"{pred.prediction} double, {pred.pred_contrib} array<double>"

        return True, "double"

    def _get_predict_func(self) -> Callable:
        """Return the true prediction function which will be running on the executor side"""

        predict_params = self._gen_predict_params_dict()
        pred_contrib_col_name = self._get_pred_contrib_col_name()

        def _predict(
            model: XGBModel, X: ArrayLike, base_margin: Optional[ArrayLike]
        ) -> Union[pd.DataFrame, pd.Series]:
            data = {}
            preds = model.predict(
                X,
                base_margin=base_margin,
                validate_features=False,
                **predict_params,
            )
            data[pred.prediction] = pd.Series(preds)

            if pred_contrib_col_name is not None:
                contribs = pred_contribs(model, X, base_margin)
                data[pred.pred_contrib] = pd.Series(list(contribs))
                return pd.DataFrame(data=data)

            return data[pred.prediction]

        return _predict

    def _post_transform(self, dataset: DataFrame, pred_col: Column) -> DataFrame:
        """Post process of transform"""
        prediction_col_name = self.getOrDefault(self.predictionCol)
        single_pred, _ = self._out_schema()

        if single_pred:
            if prediction_col_name:
                dataset = dataset.withColumn(prediction_col_name, pred_col)
        else:
            pred_struct_col = "_prediction_struct"
            dataset = dataset.withColumn(pred_struct_col, pred_col)

            if prediction_col_name:
                dataset = dataset.withColumn(
                    prediction_col_name, getattr(col(pred_struct_col), pred.prediction)
                )

            pred_contrib_col_name = self._get_pred_contrib_col_name()
            if pred_contrib_col_name is not None:
                dataset = dataset.withColumn(
                    pred_contrib_col_name,
                    array_to_vector(getattr(col(pred_struct_col), pred.pred_contrib)),
                )

            dataset = dataset.drop(pred_struct_col)
        return dataset

    def _run_on_gpu(self) -> bool:
        """If gpu is used to do the prediction according to the parameters
        and spark configurations"""

        use_gpu_by_params = super()._run_on_gpu()

        if _is_local(_get_spark_session().sparkContext):
            # if it's local model, no need to check the spark configurations
            return use_gpu_by_params

        gpu_per_task = (
            _get_spark_session()
            .sparkContext.getConf()
            .get("spark.task.resource.gpu.amount")
        )

        # User don't set gpu configurations, just use cpu
        if gpu_per_task is None:
            if use_gpu_by_params:
                get_logger(_LOG_TAG).warning(
                    "Do the prediction on the CPUs since "
                    "no gpu configurations are set"
                )
            return False

        # User already sets the gpu configurations.
        return use_gpu_by_params

    def _transform(self, dataset: DataFrame) -> DataFrame:
        # pylint: disable=too-many-statements, too-many-locals
        # Save xgb_sklearn_model and predict_params to be local variable
        # to avoid the `self` object to be pickled to remote.
        xgb_sklearn_model = self._xgb_sklearn_model

        base_margin_col = None
        if (
            self.isDefined(self.base_margin_col)
            and self.getOrDefault(self.base_margin_col) != ""
        ):
            base_margin_col = col(self.getOrDefault(self.base_margin_col)).alias(
                alias.margin
            )
        has_base_margin = base_margin_col is not None

        features_col, feature_col_names = self._get_feature_col(dataset)
        enable_sparse_data_optim = self.getOrDefault(self.enable_sparse_data_optim)

        predict_func = self._get_predict_func()

        _, schema = self._out_schema()

        is_local = _is_local(_get_spark_session().sparkContext)
        run_on_gpu = self._run_on_gpu()

        log_level = get_logger_level(_LOG_TAG)

        @pandas_udf(schema)  # type: ignore
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            assert xgb_sklearn_model is not None
            model = xgb_sklearn_model

            from pyspark import TaskContext

            context = TaskContext.get()
            assert context is not None

            dev_ordinal = -1

            msg = "Do the inference on the CPUs"
            if run_on_gpu:
                if is_cudf_available() and is_cupy_available():
                    if is_local:
                        import cupy as cp  # pylint: disable=import-error

                        total_gpus = cp.cuda.runtime.getDeviceCount()
                        if total_gpus > 0:
                            partition_id = context.partitionId()
                            # For transform local mode, default the dev_ordinal to
                            # (partition id) % gpus.
                            dev_ordinal = partition_id % total_gpus
                    else:
                        dev_ordinal = _get_gpu_id(context)

                    if dev_ordinal >= 0:
                        device = "cuda:" + str(dev_ordinal)
                        msg = "Do the inference with device: " + device
                        model.set_params(device=device)
                    else:
                        msg = "Couldn't get the correct gpu id, fallback the inference on the CPUs"
                else:
                    msg = "CUDF or Cupy is unavailable, fallback the inference on the CPUs"

            if context.partitionId() == 0:
                get_logger(_LOG_TAG, log_level).info(msg)

            def to_gpu_if_possible(data: ArrayLike) -> ArrayLike:
                """Move the data to gpu if possible"""
                if dev_ordinal >= 0:
                    import cudf  # pylint: disable=import-error
                    import cupy as cp  # pylint: disable=import-error

                    # We must set the device after import cudf, which will change the device id to 0
                    # See https://github.com/rapidsai/cudf/issues/11386
                    cp.cuda.runtime.setDevice(dev_ordinal)  # pylint: disable=I1101
                    df = cudf.DataFrame(data)
                    del data
                    return df
                return data

            for data in iterator:
                if enable_sparse_data_optim:
                    X = _read_csr_matrix_from_unwrapped_spark_vec(data)
                else:
                    if feature_col_names is not None:
                        tmp = data[feature_col_names]
                    else:
                        tmp = stack_series(data[alias.data])
                    X = to_gpu_if_possible(tmp)

                if has_base_margin:
                    base_margin = to_gpu_if_possible(data[alias.margin])
                else:
                    base_margin = None

                yield predict_func(model, X, base_margin)

        if has_base_margin:
            assert base_margin_col is not None
            pred_col = predict_udf(struct(*features_col, base_margin_col))
        else:
            pred_col = predict_udf(struct(*features_col))

        return self._post_transform(dataset, pred_col)


class _ClassificationModel(  # pylint: disable=abstract-method
    _SparkXGBModel, HasProbabilityCol, HasRawPredictionCol, HasContribPredictionCol
):
    """
    The model returned by :func:`xgboost.spark.SparkXGBClassifier.fit`

    .. Note:: This API is experimental.
    """

    def _out_schema(self) -> Tuple[bool, str]:
        schema = (
            f"{pred.raw_prediction} array<double>, {pred.prediction} double,"
            f" {pred.probability} array<double>"
        )
        if self._get_pred_contrib_col_name() is not None:
            # We will force setting strict_shape to True when predicting contribs,
            # So, it will also output 3-D shape result.
            schema = f"{schema}, {pred.pred_contrib} array<array<double>>"

        return False, schema

    def _get_predict_func(self) -> Callable:
        predict_params = self._gen_predict_params_dict()
        pred_contrib_col_name = self._get_pred_contrib_col_name()

        def transform_margin(margins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if margins.ndim == 1:
                # binomial case
                classone_probs = expit(margins)
                classzero_probs = 1.0 - classone_probs
                raw_preds = np.vstack((-margins, margins)).transpose()
                class_probs = np.vstack((classzero_probs, classone_probs)).transpose()
            else:
                # multinomial case
                raw_preds = margins
                class_probs = softmax(raw_preds, axis=1)
            return raw_preds, class_probs

        def _predict(
            model: XGBModel, X: ArrayLike, base_margin: Optional[np.ndarray]
        ) -> Union[pd.DataFrame, pd.Series]:
            margins = model.predict(
                X,
                base_margin=base_margin,
                output_margin=True,
                validate_features=False,
                **predict_params,
            )
            raw_preds, class_probs = transform_margin(margins)

            # It seems that they use argmax of class probs,
            # not of margin to get the prediction (Note: scala implementation)
            preds = np.argmax(class_probs, axis=1)
            result: Dict[str, pd.Series] = {
                pred.raw_prediction: pd.Series(list(raw_preds)),
                pred.prediction: pd.Series(preds),
                pred.probability: pd.Series(list(class_probs)),
            }

            if pred_contrib_col_name is not None:
                contribs = pred_contribs(model, X, base_margin, strict_shape=True)
                result[pred.pred_contrib] = pd.Series(list(contribs.tolist()))

            return pd.DataFrame(data=result)

        return _predict

    def _post_transform(self, dataset: DataFrame, pred_col: Column) -> DataFrame:
        pred_struct_col = "_prediction_struct"
        dataset = dataset.withColumn(pred_struct_col, pred_col)

        raw_prediction_col_name = self.getOrDefault(self.rawPredictionCol)
        if raw_prediction_col_name:
            dataset = dataset.withColumn(
                raw_prediction_col_name,
                array_to_vector(getattr(col(pred_struct_col), pred.raw_prediction)),
            )

        prediction_col_name = self.getOrDefault(self.predictionCol)
        if prediction_col_name:
            dataset = dataset.withColumn(
                prediction_col_name, getattr(col(pred_struct_col), pred.prediction)
            )

        probability_col_name = self.getOrDefault(self.probabilityCol)
        if probability_col_name:
            dataset = dataset.withColumn(
                probability_col_name,
                array_to_vector(getattr(col(pred_struct_col), pred.probability)),
            )

        pred_contrib_col_name = self._get_pred_contrib_col_name()
        if pred_contrib_col_name is not None:
            dataset = dataset.withColumn(
                pred_contrib_col_name,
                getattr(col(pred_struct_col), pred.pred_contrib),
            )

        return dataset.drop(pred_struct_col)


class _SparkXGBSharedReadWrite:
    @staticmethod
    def saveMetadata(
        instance: Union[_SparkXGBEstimator, _SparkXGBModel],
        path: str,
        sc: SparkContext,
        logger: logging.Logger,
        extraMetadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save the metadata of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.
        """
        instance._validate_params()
        skipParams = ["callbacks", "xgb_model"]
        jsonParams = {}
        for p, v in instance._paramMap.items():  # pylint: disable=protected-access
            if p.name not in skipParams:
                jsonParams[p.name] = v

        extraMetadata = extraMetadata or {}
        callbacks = instance.getOrDefault("callbacks")
        if callbacks is not None:
            logger.warning(
                "The callbacks parameter is saved using cloudpickle and it "
                "is not a fully self-contained format. It may fail to load "
                "with different versions of dependencies."
            )
            serialized_callbacks = base64.encodebytes(
                cloudpickle.dumps(callbacks)
            ).decode("ascii")
            extraMetadata["serialized_callbacks"] = serialized_callbacks
        init_booster = instance.getOrDefault("xgb_model")
        if init_booster is not None:
            extraMetadata["init_booster"] = _INIT_BOOSTER_SAVE_PATH
        DefaultParamsWriter.saveMetadata(
            instance, path, sc, extraMetadata=extraMetadata, paramMap=jsonParams
        )
        if init_booster is not None:
            ser_init_booster = serialize_booster(init_booster)
            save_path = os.path.join(path, _INIT_BOOSTER_SAVE_PATH)
            _get_spark_session().createDataFrame(
                [(ser_init_booster,)], ["init_booster"]
            ).write.parquet(save_path)

    @staticmethod
    def loadMetadataAndInstance(
        pyspark_xgb_cls: Union[Type[_SparkXGBEstimator], Type[_SparkXGBModel]],
        path: str,
        sc: SparkContext,
        logger: logging.Logger,
    ) -> Tuple[Dict[str, Any], Union[_SparkXGBEstimator, _SparkXGBModel]]:
        """
        Load the metadata and the instance of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.

        :return: a tuple of (metadata, instance)
        """
        metadata = DefaultParamsReader.loadMetadata(
            path, sc, expectedClassName=get_class_name(pyspark_xgb_cls)
        )
        pyspark_xgb = pyspark_xgb_cls()
        DefaultParamsReader.getAndSetParams(pyspark_xgb, metadata)

        if "serialized_callbacks" in metadata:
            serialized_callbacks = metadata["serialized_callbacks"]
            try:
                callbacks = cloudpickle.loads(
                    base64.decodebytes(serialized_callbacks.encode("ascii"))
                )
                pyspark_xgb.set(pyspark_xgb.callbacks, callbacks)  # type: ignore
            except Exception as e:  # pylint: disable=W0703
                logger.warning(
                    f"Fails to load the callbacks param due to {e}. Please set the "
                    "callbacks param manually for the loaded estimator."
                )

        if "init_booster" in metadata:
            load_path = os.path.join(path, metadata["init_booster"])
            ser_init_booster = (
                _get_spark_session().read.parquet(load_path).collect()[0].init_booster
            )
            init_booster = deserialize_booster(ser_init_booster)
            pyspark_xgb.set(pyspark_xgb.xgb_model, init_booster)  # type: ignore

        pyspark_xgb._resetUid(metadata["uid"])  # pylint: disable=protected-access
        return metadata, pyspark_xgb


class SparkXGBWriter(MLWriter):
    """
    Spark Xgboost estimator writer.
    """

    def __init__(self, instance: "_SparkXGBEstimator") -> None:
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def saveImpl(self, path: str) -> None:
        """
        save model.
        """
        _SparkXGBSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)


class SparkXGBReader(MLReader):
    """
    Spark Xgboost estimator reader.
    """

    def __init__(self, cls: Type["_SparkXGBEstimator"]) -> None:
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def load(self, path: str) -> "_SparkXGBEstimator":
        """
        load model.
        """
        _, pyspark_xgb = _SparkXGBSharedReadWrite.loadMetadataAndInstance(
            self.cls, path, self.sc, self.logger
        )
        return cast("_SparkXGBEstimator", pyspark_xgb)


class SparkXGBModelWriter(MLWriter):
    """
    Spark Xgboost model writer.
    """

    def __init__(self, instance: _SparkXGBModel) -> None:
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def saveImpl(self, path: str) -> None:
        """
        Save metadata and model for a :py:class:`_SparkXGBModel`
        - save metadata to path/metadata
        - save model to path/model.json
        """
        xgb_model = self.instance._xgb_sklearn_model
        assert xgb_model is not None
        _SparkXGBSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)
        model_save_path = os.path.join(path, "model")
        booster = xgb_model.get_booster().save_raw("json").decode("utf-8")
        booster_chunks = []

        for offset in range(0, len(booster), _MODEL_CHUNK_SIZE):
            booster_chunks.append(booster[offset : offset + _MODEL_CHUNK_SIZE])

        _get_spark_session().sparkContext.parallelize(booster_chunks, 1).saveAsTextFile(
            model_save_path
        )


class SparkXGBModelReader(MLReader):
    """
    Spark Xgboost model reader.
    """

    def __init__(self, cls: Type["_SparkXGBModel"]) -> None:
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def load(self, path: str) -> "_SparkXGBModel":
        """
        Load metadata and model for a :py:class:`_SparkXGBModel`

        :return: SparkXGBRegressorModel or SparkXGBClassifierModel instance
        """
        _, py_model = _SparkXGBSharedReadWrite.loadMetadataAndInstance(
            self.cls, path, self.sc, self.logger
        )
        py_model = cast("_SparkXGBModel", py_model)

        xgb_sklearn_params = py_model._gen_xgb_params_dict(
            gen_xgb_sklearn_estimator_param=True
        )
        model_load_path = os.path.join(path, "model")

        ser_xgb_model = "".join(
            _get_spark_session().sparkContext.textFile(model_load_path).collect()
        )

        def create_xgb_model() -> "XGBModel":
            return self.cls._xgb_cls()(**xgb_sklearn_params)

        xgb_model = deserialize_xgb_model(ser_xgb_model, create_xgb_model)
        py_model._xgb_sklearn_model = xgb_model
        return py_model
