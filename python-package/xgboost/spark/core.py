# type: ignore
"""Xgboost pyspark integration submodule for core code."""
# pylint: disable=fixme, too-many-ancestors, protected-access, no-member, invalid-name
# pylint: disable=too-few-public-methods
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
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
from pyspark.ml.util import MLReadable, MLWritable
from pyspark.sql.functions import col, countDistinct, pandas_udf, struct
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
from xgboost.core import Booster
from xgboost.training import train as worker_train

import xgboost
from xgboost import XGBClassifier, XGBRegressor

from .data import alias, create_dmatrix_from_partitions, stack_series
from .model import (
    SparkXGBModelReader,
    SparkXGBModelWriter,
    SparkXGBReader,
    SparkXGBWriter,
)
from .params import HasArbitraryParamsDict, HasBaseMarginCol
from .utils import (
    RabitContext,
    _get_args_from_message_list,
    _get_default_params_from_func,
    _get_gpu_id,
    _get_max_num_concurrent_tasks,
    _get_rabit_args,
    _get_spark_session,
    _is_local,
    get_class_name,
    get_logger,
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
    "use_gpu",
    "feature_names",
]

_non_booster_params = [
    "missing",
    "n_estimators",
    "feature_types",
    "feature_weights",
]

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
    "gpu_id",  # we have "use_gpu" pyspark param instead.
    "enable_categorical",  # Use feature_types param to specify categorical feature instead
    "use_label_encoder",
    "n_jobs",  # Do not allow user to set it, will use `spark.task.cpus` value instead.
    "nthread",  # Ditto
]

_unsupported_fit_params = {
    "sample_weight",  # Supported by spark param weightCol
    # Supported by spark param weightCol # and validationIndicatorCol
    "eval_set",
    "sample_weight_eval_set",
    "base_margin",  # Supported by spark param base_margin_col
}

_unsupported_predict_params = {
    # for classification, we can use rawPrediction as margin
    "output_margin",
    "validate_features",  # TODO
    "base_margin",  # Use pyspark base_margin_col param instead.
}


class _SparkXGBParams(
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
    HasPredictionCol,
    HasValidationIndicatorCol,
    HasArbitraryParamsDict,
    HasBaseMarginCol,
):
    num_workers = Param(
        Params._dummy(),
        "num_workers",
        "The number of XGBoost workers. Each XGBoost worker corresponds to one spark task.",
        TypeConverters.toInt,
    )
    use_gpu = Param(
        Params._dummy(),
        "use_gpu",
        "A boolean variable. Set use_gpu=true if the executors "
        + "are running on GPU instances. Currently, only one GPU per task is supported.",
    )
    force_repartition = Param(
        Params._dummy(),
        "force_repartition",
        "A boolean variable. Set force_repartition=true if you "
        + "want to force the input dataset to be repartitioned before XGBoost training."
        + "Note: The auto repartitioning judgement is not fully accurate, so it is recommended"
        + "to have force_repartition be True.",
    )
    feature_names = Param(
        Params._dummy(), "feature_names", "A list of str to specify feature names."
    )

    @classmethod
    def _xgb_cls(cls):
        """
        Subclasses should override this method and
        returns an xgboost.XGBModel subclass
        """
        raise NotImplementedError()

    # Parameters for xgboost.XGBModel()
    @classmethod
    def _get_xgb_params_default(cls):
        xgb_model_default = cls._xgb_cls()()
        params_dict = xgb_model_default.get_params()
        filtered_params_dict = {
            k: params_dict[k] for k in params_dict if k not in _unsupported_xgb_params
        }
        return filtered_params_dict

    def _set_xgb_params_default(self):
        filtered_params_dict = self._get_xgb_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_xgb_params_dict(self, gen_xgb_sklearn_estimator_param=False):
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
    def _get_fit_params_default(cls):
        fit_params = _get_default_params_from_func(
            cls._xgb_cls().fit, _unsupported_fit_params
        )
        return fit_params

    def _set_fit_params_default(self):
        filtered_params_dict = self._get_fit_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_fit_params_dict(self):
        """
        Returns a dict of params for .fit()
        """
        fit_params_keys = self._get_fit_params_default().keys()
        fit_params = {}
        for param in self.extractParamMap():
            if param.name in fit_params_keys:
                fit_params[param.name] = self.getOrDefault(param)
        return fit_params

    # Parameters for xgboost.XGBModel().predict()
    @classmethod
    def _get_predict_params_default(cls):
        predict_params = _get_default_params_from_func(
            cls._xgb_cls().predict, _unsupported_predict_params
        )
        return predict_params

    def _set_predict_params_default(self):
        filtered_params_dict = self._get_predict_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_predict_params_dict(self):
        """
        Returns a dict of params for .predict()
        """
        predict_params_keys = self._get_predict_params_default().keys()
        predict_params = {}
        for param in self.extractParamMap():
            if param.name in predict_params_keys:
                predict_params[param.name] = self.getOrDefault(param)
        return predict_params

    def _validate_params(self):
        init_model = self.getOrDefault(self.xgb_model)
        if init_model is not None:
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

        if (
            self.getOrDefault(self.force_repartition)
            and self.getOrDefault(self.num_workers) == 1
        ):
            get_logger(self.__class__.__name__).warning(
                "You set force_repartition to true when there is no need for a repartition."
                "Therefore, that parameter will be ignored."
            )

        if self.getOrDefault(self.use_gpu):
            tree_method = self.getParam("tree_method")
            if (
                self.getOrDefault(tree_method) is not None
                and self.getOrDefault(tree_method) != "gpu_hist"
            ):
                raise ValueError(
                    f"tree_method should be 'gpu_hist' or None when use_gpu is True,"
                    f"found {self.getOrDefault(tree_method)}."
                )

            gpu_per_task = (
                _get_spark_session()
                .sparkContext.getConf()
                .get("spark.task.resource.gpu.amount")
            )

            is_local = _is_local(_get_spark_session().sparkContext)

            if is_local:
                # checking spark local mode.
                if gpu_per_task:
                    raise RuntimeError(
                        "The spark cluster does not support gpu configuration for local mode. "
                        "Please delete spark.executor.resource.gpu.amount and "
                        "spark.task.resource.gpu.amount"
                    )

                # Support GPU training in Spark local mode is just for debugging purposes,
                # so it's okay for printing the below warning instead of checking the real
                # gpu numbers and raising the exception.
                get_logger(self.__class__.__name__).warning(
                    "You enabled use_gpu in spark local mode. Please make sure your local node "
                    "has at least %d GPUs",
                    self.getOrDefault(self.num_workers),
                )
            else:
                # checking spark non-local mode.
                if not gpu_per_task or int(gpu_per_task) < 1:
                    raise RuntimeError(
                        "The spark cluster does not have the necessary GPU"
                        + "configuration for the spark task. Therefore, we cannot"
                        + "run xgboost training using GPU."
                    )

                if int(gpu_per_task) > 1:
                    get_logger(self.__class__.__name__).warning(
                        "You configured %s GPU cores for each spark task, but in "
                        "XGBoost training, every Spark task will only use one GPU core.",
                        gpu_per_task,
                    )


def _validate_and_convert_feature_col_as_array_col(dataset, features_col_name):
    features_col_datatype = dataset.schema[features_col_name].dataType
    features_col = col(features_col_name)
    if isinstance(features_col_datatype, ArrayType):
        if not isinstance(
            features_col_datatype.elementType,
            (DoubleType, FloatType, LongType, IntegerType, ShortType),
        ):
            raise ValueError(
                "If feature column is array type, its elements must be number type."
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


class _SparkXGBEstimator(Estimator, _SparkXGBParams, MLReadable, MLWritable):
    def __init__(self):
        super().__init__()
        self._set_xgb_params_default()
        self._set_fit_params_default()
        self._set_predict_params_default()
        # Note: The default value for arbitrary_params_dict must always be empty dict.
        #  For additional settings added into "arbitrary_params_dict" by default,
        #  they are added in `setParams`.
        self._setDefault(
            num_workers=1,
            use_gpu=False,
            force_repartition=False,
            feature_names=None,
            feature_types=None,
            arbitrary_params_dict={},
        )

    def setParams(self, **kwargs):  # pylint: disable=invalid-name
        """
        Set params for the estimator.
        """
        _extra_params = {}
        if "arbitrary_params_dict" in kwargs:
            raise ValueError("Invalid param name: 'arbitrary_params_dict'.")

        for k, v in kwargs.items():
            if k in _inverse_pyspark_param_alias_map:
                raise ValueError(
                    f"Please use param name {_inverse_pyspark_param_alias_map[k]} instead."
                )
            if k in _pyspark_param_alias_map:
                real_k = _pyspark_param_alias_map[k]
                k = real_k

            if self.hasParam(k):
                self._set(**{str(k): v})
            else:
                if (
                    k in _unsupported_xgb_params
                    or k in _unsupported_fit_params
                    or k in _unsupported_predict_params
                ):
                    raise ValueError(f"Unsupported param '{k}'.")
                _extra_params[k] = v
        _existing_extra_params = self.getOrDefault(self.arbitrary_params_dict)
        self._set(arbitrary_params_dict={**_existing_extra_params, **_extra_params})

    @classmethod
    def _pyspark_model_cls(cls):
        """
        Subclasses should override this method and
        returns a _SparkXGBModel subclass
        """
        raise NotImplementedError()

    def _create_pyspark_model(self, xgb_model):
        return self._pyspark_model_cls()(xgb_model)

    def _convert_to_sklearn_model(self, booster: bytearray, config: str):
        xgb_sklearn_params = self._gen_xgb_params_dict(
            gen_xgb_sklearn_estimator_param=True
        )
        sklearn_model = self._xgb_cls()(**xgb_sklearn_params)
        sklearn_model.load_model(booster)
        sklearn_model._Booster.load_config(config)
        return sklearn_model

    def _query_plan_contains_valid_repartition(self, dataset):
        """
        Returns true if the latest element in the logical plan is a valid repartition
        The logic plan string format is like:

        == Optimized Logical Plan ==
        Repartition 4, true
        +- LogicalRDD [features#12, label#13L], false

        i.e., the top line in the logical plan is the last operation to execute.
        so, in this method, we check the first line, if it is a "Repartition" operation,
        and the result dataframe has the same partition number with num_workers param,
        then it means the dataframe is well repartitioned and we don't need to
        repartition the dataframe again.
        """
        num_partitions = dataset.rdd.getNumPartitions()
        query_plan = dataset._sc._jvm.PythonSQLUtils.explainString(
            dataset._jdf.queryExecution(), "extended"
        )
        start = query_plan.index("== Optimized Logical Plan ==")
        start += len("== Optimized Logical Plan ==") + 1
        num_workers = self.getOrDefault(self.num_workers)
        if (
            query_plan[start : start + len("Repartition")] == "Repartition"
            and num_workers == num_partitions
        ):
            return True
        return False

    def _repartition_needed(self, dataset):
        """
        We repartition the dataset if the number of workers is not equal to the number of
        partitions. There is also a check to make sure there was "active partitioning"
        where either Round Robin or Hash partitioning was actively used before this stage.
        """
        if self.getOrDefault(self.force_repartition):
            return True
        try:
            if self._query_plan_contains_valid_repartition(dataset):
                return False
        except Exception:  # pylint: disable=broad-except
            pass
        return True

    def _get_distributed_train_params(self, dataset):
        """
        This just gets the configuration params for distributed xgboost
        """
        params = self._gen_xgb_params_dict()
        fit_params = self._gen_fit_params_dict()
        verbose_eval = fit_params.pop("verbose", None)

        params.update(fit_params)
        params["verbose_eval"] = verbose_eval
        classification = self._xgb_cls() == XGBClassifier
        num_classes = int(dataset.select(countDistinct(alias.label)).collect()[0][0])
        if classification and num_classes == 2:
            params["objective"] = "binary:logistic"
        elif classification and num_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_classes
        else:
            params["objective"] = "reg:squarederror"

        # TODO: support "num_parallel_tree" for random forest
        params["num_boost_round"] = self.getOrDefault(self.n_estimators)

        if self.getOrDefault(self.use_gpu):
            params["tree_method"] = "gpu_hist"

        return params

    @classmethod
    def _get_xgb_train_call_args(cls, train_params):
        xgb_train_default_args = _get_default_params_from_func(xgboost.train, {})
        booster_params, kwargs_params = {}, {}
        for key, value in train_params.items():
            if key in xgb_train_default_args:
                kwargs_params[key] = value
            else:
                booster_params[key] = value
        return booster_params, kwargs_params

    def _fit(self, dataset):
        # pylint: disable=too-many-statements, too-many-locals
        self._validate_params()
        label_col = col(self.getOrDefault(self.labelCol)).alias(alias.label)

        features_array_col = _validate_and_convert_feature_col_as_array_col(
            dataset, self.getOrDefault(self.featuresCol)
        )
        select_cols = [features_array_col, label_col]

        if self.isDefined(self.weightCol) and self.getOrDefault(self.weightCol):
            select_cols.append(
                col(self.getOrDefault(self.weightCol)).alias(alias.weight)
            )

        if self.isDefined(self.validationIndicatorCol) and self.getOrDefault(
            self.validationIndicatorCol
        ):
            select_cols.append(
                col(self.getOrDefault(self.validationIndicatorCol)).alias(alias.valid)
            )

        if self.isDefined(self.base_margin_col) and self.getOrDefault(
            self.base_margin_col
        ):
            select_cols.append(
                col(self.getOrDefault(self.base_margin_col)).alias(alias.margin)
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
            dataset = dataset.repartition(num_workers)
        train_params = self._get_distributed_train_params(dataset)
        booster_params, train_call_kwargs_params = self._get_xgb_train_call_args(
            train_params
        )

        cpu_per_task = int(
            _get_spark_session().sparkContext.getConf().get("spark.task.cpus", "1")
        )

        dmatrix_kwargs = {
            "nthread": cpu_per_task,
            "feature_types": self.getOrDefault(self.feature_types),
            "feature_names": self.getOrDefault(self.feature_names),
            "feature_weights": self.getOrDefault(self.feature_weights),
            "missing": self.getOrDefault(self.missing),
        }
        booster_params["nthread"] = cpu_per_task
        use_gpu = self.getOrDefault(self.use_gpu)

        is_local = _is_local(_get_spark_session().sparkContext)

        def _train_booster(pandas_df_iter):
            """Takes in an RDD partition and outputs a booster for that partition after
            going through the Rabit Ring protocol

            """
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            context.barrier()

            if use_gpu:
                booster_params["gpu_id"] = (
                    context.partitionId() if is_local else _get_gpu_id(context)
                )

            _rabit_args = ""
            if context.partitionId() == 0:
                _rabit_args = str(_get_rabit_args(context, num_workers))

            messages = context.allGather(message=str(_rabit_args))
            _rabit_args = _get_args_from_message_list(messages)
            evals_result = {}
            with RabitContext(_rabit_args, context):
                dtrain, dvalid = create_dmatrix_from_partitions(
                    pandas_df_iter,
                    None,
                    dmatrix_kwargs,
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
                yield pd.DataFrame(
                    data={
                        "config": [booster.save_config()],
                        "booster": [booster.save_raw("json").decode("utf-8")],
                    }
                )

        def _run_job():
            ret = (
                dataset.mapInPandas(
                    _train_booster, schema="config string, booster string"
                )
                .rdd.barrier()
                .mapPartitions(lambda x: x)
                .collect()[0]
            )
            return ret[0], ret[1]

        (config, booster) = _run_job()

        result_xgb_model = self._convert_to_sklearn_model(
            bytearray(booster, "utf-8"), config
        )
        return self._copyValues(self._create_pyspark_model(result_xgb_model))

    def write(self):
        """
        Return the writer for saving the estimator.
        """
        return SparkXGBWriter(self)

    @classmethod
    def read(cls):
        """
        Return the reader for loading the estimator.
        """
        return SparkXGBReader(cls)


class _SparkXGBModel(Model, _SparkXGBParams, MLReadable, MLWritable):
    def __init__(self, xgb_sklearn_model=None):
        super().__init__()
        self._xgb_sklearn_model = xgb_sklearn_model

    def get_booster(self):
        """
        Return the `xgboost.core.Booster` instance.
        """
        return self._xgb_sklearn_model.get_booster()

    def get_feature_importances(self, importance_type="weight"):
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

    def write(self):
        """
        Return the writer for saving the model.
        """
        return SparkXGBModelWriter(self)

    @classmethod
    def read(cls):
        """
        Return the reader for loading the model.
        """
        return SparkXGBModelReader(cls)

    def _transform(self, dataset):
        raise NotImplementedError()


class SparkXGBRegressorModel(_SparkXGBModel):
    """
    The model returned by :func:`xgboost.spark.SparkXGBRegressor.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls):
        return XGBRegressor

    def _transform(self, dataset):
        # Save xgb_sklearn_model and predict_params to be local variable
        # to avoid the `self` object to be pickled to remote.
        xgb_sklearn_model = self._xgb_sklearn_model
        predict_params = self._gen_predict_params_dict()

        has_base_margin = False
        if self.isDefined(self.base_margin_col) and self.getOrDefault(
            self.base_margin_col
        ):
            has_base_margin = True
            base_margin_col = col(self.getOrDefault(self.base_margin_col)).alias(
                alias.margin
            )

        @pandas_udf("double")
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            model = xgb_sklearn_model
            for data in iterator:
                X = stack_series(data[alias.data])
                if has_base_margin:
                    base_margin = data[alias.margin].to_numpy()
                else:
                    base_margin = None

                preds = model.predict(
                    X,
                    base_margin=base_margin,
                    validate_features=False,
                    **predict_params,
                )
                yield pd.Series(preds)

        features_col = _validate_and_convert_feature_col_as_array_col(
            dataset, self.getOrDefault(self.featuresCol)
        )

        if has_base_margin:
            pred_col = predict_udf(struct(features_col, base_margin_col))
        else:
            pred_col = predict_udf(struct(features_col))

        predictionColName = self.getOrDefault(self.predictionCol)

        return dataset.withColumn(predictionColName, pred_col)


class SparkXGBClassifierModel(_SparkXGBModel, HasProbabilityCol, HasRawPredictionCol):
    """
    The model returned by :func:`xgboost.spark.SparkXGBClassifier.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls):
        return XGBClassifier

    def _transform(self, dataset):
        # Save xgb_sklearn_model and predict_params to be local variable
        # to avoid the `self` object to be pickled to remote.
        xgb_sklearn_model = self._xgb_sklearn_model
        predict_params = self._gen_predict_params_dict()

        has_base_margin = False
        if self.isDefined(self.base_margin_col) and self.getOrDefault(
            self.base_margin_col
        ):
            has_base_margin = True
            base_margin_col = col(self.getOrDefault(self.base_margin_col)).alias(
                alias.margin
            )

        def transform_margin(margins: np.ndarray):
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

        @pandas_udf(
            "rawPrediction array<double>, prediction double, probability array<double>"
        )
        def predict_udf(
            iterator: Iterator[Tuple[pd.Series, ...]]
        ) -> Iterator[pd.DataFrame]:
            model = xgb_sklearn_model
            for data in iterator:
                X = stack_series(data[alias.data])
                if has_base_margin:
                    base_margin = stack_series(data[alias.margin])
                else:
                    base_margin = None

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
                yield pd.DataFrame(
                    data={
                        "rawPrediction": pd.Series(list(raw_preds)),
                        "prediction": pd.Series(preds),
                        "probability": pd.Series(list(class_probs)),
                    }
                )

        features_col = _validate_and_convert_feature_col_as_array_col(
            dataset, self.getOrDefault(self.featuresCol)
        )

        if has_base_margin:
            pred_struct = predict_udf(struct(features_col, base_margin_col))
        else:
            pred_struct = predict_udf(struct(features_col))

        pred_struct_col = "_prediction_struct"

        rawPredictionColName = self.getOrDefault(self.rawPredictionCol)
        predictionColName = self.getOrDefault(self.predictionCol)
        probabilityColName = self.getOrDefault(self.probabilityCol)
        dataset = dataset.withColumn(pred_struct_col, pred_struct)
        if rawPredictionColName:
            dataset = dataset.withColumn(
                rawPredictionColName,
                array_to_vector(col(pred_struct_col).rawPrediction),
            )
        if predictionColName:
            dataset = dataset.withColumn(
                predictionColName, col(pred_struct_col).prediction
            )
        if probabilityColName:
            dataset = dataset.withColumn(
                probabilityColName, array_to_vector(col(pred_struct_col).probability)
            )

        return dataset.drop(pred_struct_col)


def _set_pyspark_xgb_cls_param_attrs(pyspark_estimator_class, pyspark_model_class):
    params_dict = pyspark_estimator_class._get_xgb_params_default()

    def param_value_converter(v):
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        if isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        if isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        return v

    def set_param_attrs(attr_name, param_obj_):
        param_obj_.typeConverter = param_value_converter
        setattr(pyspark_estimator_class, attr_name, param_obj_)
        setattr(pyspark_model_class, attr_name, param_obj_)

    for name in params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of "
            f"{get_class_name(pyspark_estimator_class._xgb_cls())} for this param {name}"
        )

        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    fit_params_dict = pyspark_estimator_class._get_fit_params_default()
    for name in fit_params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}"
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

    predict_params_dict = pyspark_estimator_class._get_predict_params_default()
    for name in predict_params_dict.keys():
        doc = (
            f"Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}"
            f".predict() for this param {name}"
        )
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)
