import shutil
import tempfile
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasWeightCol, \
    HasPredictionCol, HasProbabilityCol, HasRawPredictionCol, HasValidationIndicatorCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import MLReadable, MLWritable
from pyspark.sql.functions import col, pandas_udf, countDistinct, struct
from pyspark.sql.types import ArrayType, FloatType
from xgboost import XGBClassifier, XGBRegressor
from xgboost.core import Booster
import cloudpickle
import xgboost
from xgboost.training import train as worker_train
from .utils import get_logger, _get_max_num_concurrent_tasks
from .data import prepare_predict_data, prepare_train_val_data, convert_partition_data_to_dmatrix
from .model import (XgboostReader, XgboostWriter, XgboostModelReader,
                    XgboostModelWriter, deserialize_xgb_model,
                    get_xgb_model_creator, serialize_xgb_model)
from .utils import (_get_default_params_from_func, get_class_name,
                    HasArbitraryParamsDict, HasBaseMarginCol, RabitContext,
                    _get_rabit_args, _get_args_from_message_list,
                    _get_spark_session)

from pyspark.ml.functions import array_to_vector, vector_to_array

# Put pyspark specific params here, they won't be passed to XGBoost.
# like `validationIndicatorCol`, `baseMarginCol`
_pyspark_specific_params = [
    'featuresCol', 'labelCol', 'weightCol', 'rawPredictionCol',
    'predictionCol', 'probabilityCol', 'validationIndicatorCol'
                                       'baseMarginCol'
]

_unsupported_xgb_params = [
    'gpu_id',  # we have "use_gpu" pyspark param instead.
]
_unsupported_fit_params = {
    'sample_weight',  # Supported by spark param weightCol
    # Supported by spark param weightCol # and validationIndicatorCol
    'eval_set',
    'sample_weight_eval_set',
    'base_margin'  # Supported by spark param baseMarginCol
}
_unsupported_predict_params = {
    # for classification, we can use rawPrediction as margin
    'output_margin',
    'validate_features',  # TODO
    'base_margin'  # TODO
}

_created_params = {"num_workers", "use_gpu"}


class _XgboostParams(HasFeaturesCol, HasLabelCol, HasWeightCol,
                     HasPredictionCol, HasValidationIndicatorCol,
                     HasArbitraryParamsDict, HasBaseMarginCol):
    num_workers = Param(
        Params._dummy(), "num_workers",
        "The number of XGBoost workers. Each XGBoost worker corresponds to one spark task.",
        TypeConverters.toInt)
    use_gpu = Param(
        Params._dummy(), "use_gpu",
        "A boolean variable. Set use_gpu=true if the executors " +
        "are running on GPU instances. Currently, only one GPU per task is supported."
    )
    force_repartition = Param(
        Params._dummy(), "force_repartition",
        "A boolean variable. Set force_repartition=true if you " +
        "want to force the input dataset to be repartitioned before XGBoost training." +
        "Note: The auto repartitioning judgement is not fully accurate, so it is recommended" +
        "to have force_repartition be True.")
    use_external_storage = Param(
        Params._dummy(), "use_external_storage",
        "A boolean variable (that is False by default). External storage is a parameter" +
        "for distributed training that allows external storage (disk) to be used when." +
        "you have an exceptionally large dataset. This should be set to false for" +
        "small datasets. Note that base margin and weighting doesn't work if this is True." +
        "Also note that you may use precision if you use external storage."
    )
    external_storage_precision = Param(
        Params._dummy(), "external_storage_precision",
        "The number of significant digits for data storage on disk when using external storage.",
        TypeConverters.toInt
    )

    @classmethod
    def _xgb_cls(cls):
        """
        Subclasses should override this method and
        returns an xgboost.XGBModel subclass
        """
        raise NotImplementedError()

    def _get_xgb_model_creator(self):
        arbitaryParamsDict = self.getOrDefault(
            self.getParam("arbitraryParamsDict"))
        total_params = {**self._gen_xgb_params_dict(), **arbitaryParamsDict}
        # Once we have already added all of the elements of kwargs, we can just remove it
        del total_params["arbitraryParamsDict"]
        for param in _created_params:
            del total_params[param]
        return get_xgb_model_creator(self._xgb_cls(), total_params)

    # Parameters for xgboost.XGBModel()
    @classmethod
    def _get_xgb_params_default(cls):
        xgb_model_default = cls._xgb_cls()()
        params_dict = xgb_model_default.get_params()
        filtered_params_dict = {
            k: params_dict[k]
            for k in params_dict if k not in _unsupported_xgb_params
        }
        return filtered_params_dict

    def _set_xgb_params_default(self):
        filtered_params_dict = self._get_xgb_params_default()
        self._setDefault(**filtered_params_dict)
        self._setDefault(**{"arbitraryParamsDict": {}})

    def _gen_xgb_params_dict(self):
        xgb_params = {}
        non_xgb_params = \
            set(_pyspark_specific_params) | \
            self._get_fit_params_default().keys() | \
            self._get_predict_params_default().keys()
        for param in self.extractParamMap():
            if param.name not in non_xgb_params:
                xgb_params[param.name] = self.getOrDefault(param)
        return xgb_params

    def _set_distributed_params(self):
        self.set(self.num_workers, 1)
        self.set(self.use_gpu, False)
        self.set(self.force_repartition, False)
        self.set(self.use_external_storage, False)
        self.set(self.external_storage_precision, 5)  # Check if this needs to be modified

    # Parameters for xgboost.XGBModel().fit()
    @classmethod
    def _get_fit_params_default(cls):
        fit_params = _get_default_params_from_func(cls._xgb_cls().fit,
                                                   _unsupported_fit_params)
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
            cls._xgb_cls().predict, _unsupported_predict_params)
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
                    'The xgb_model param must be set with a `xgboost.core.Booster` '
                    'instance.')

        if self.getOrDefault(self.num_workers) < 1:
            raise ValueError(
                f"Number of workers was {self.getOrDefault(self.num_workers)}."
                f"It cannot be less than 1 [Default is 1]")

        if self.getOrDefault(self.num_workers) > 1 and not self.getOrDefault(
                self.use_gpu):
            cpu_per_task = _get_spark_session().sparkContext.getConf().get(
                'spark.task.cpus')
            if cpu_per_task and int(cpu_per_task) > 1:
                get_logger(self.__class__.__name__).warning(
                    f'You configured {cpu_per_task} CPU cores for each spark task, but in '
                    f'XGBoost training, every Spark task will only use one CPU core.'
                )

        if self.getOrDefault(self.force_repartition) and self.getOrDefault(
                self.num_workers) == 1:
            get_logger(self.__class__.__name__).warning(
                "You set force_repartition to true when there is no need for a repartition."
                "Therefore, that parameter will be ignored.")

        if self.getOrDefault(self.use_gpu):
            tree_method = self.getParam("tree_method")
            if self.getOrDefault(
                    tree_method
            ) is not None and self.getOrDefault(tree_method) != "gpu_hist":
                raise ValueError(
                    f"tree_method should be 'gpu_hist' or None when use_gpu is True,"
                    f"found {self.getOrDefault(tree_method)}.")

            gpu_per_task = _get_spark_session().sparkContext.getConf().get(
                'spark.task.resource.gpu.amount')

            if not gpu_per_task or int(gpu_per_task) < 1:
                raise RuntimeError(
                    "The spark cluster does not have the necessary GPU" +
                    "configuration for the spark task. Therefore, we cannot" +
                    "run xgboost training using GPU.")

            if int(gpu_per_task) > 1:
                get_logger(self.__class__.__name__).warning(
                    f'You configured {gpu_per_task} GPU cores for each spark task, but in '
                    f'XGBoost training, every Spark task will only use one GPU core.'
                )


class _XgboostEstimator(Estimator, _XgboostParams, MLReadable, MLWritable):
    def __init__(self):
        super().__init__()
        self._set_xgb_params_default()
        self._set_fit_params_default()
        self._set_predict_params_default()
        self._set_distributed_params()

    def setParams(self, **kwargs):
        _user_defined = {}
        for k, v in kwargs.items():
            if self.hasParam(k):
                self._set(**{str(k): v})
            else:
                _user_defined[k] = v
        _defined_args = self.getOrDefault(self.getParam("arbitraryParamsDict"))
        _defined_args.update(_user_defined)
        self._set(**{"arbitraryParamsDict": _defined_args})

    @classmethod
    def _pyspark_model_cls(cls):
        """
        Subclasses should override this method and
        returns a _XgboostModel subclass
        """
        raise NotImplementedError()

    def _create_pyspark_model(self, xgb_model):
        return self._pyspark_model_cls()(xgb_model)

    @classmethod
    def _convert_to_classifier(cls, booster):
        clf = XGBClassifier()
        clf._Booster = booster
        return clf

    @classmethod
    def _convert_to_regressor(cls, booster):
        reg = XGBRegressor()
        reg._Booster = booster
        return reg

    def _convert_to_model(self, booster):
        if self._xgb_cls() == XGBRegressor:
            return self._convert_to_regressor(booster)
        elif self._xgb_cls() == XGBClassifier:
            return self._convert_to_classifier(booster)
        else:
            return None  # check if this else statement is needed.

    def _query_plan_contains_valid_repartition(self, query_plan,
                                               num_partitions):
        """
        Returns true if the latest element in the logical plan is a valid repartition
        """
        start = query_plan.index("== Optimized Logical Plan ==")
        start += len("== Optimized Logical Plan ==") + 1
        num_workers = self.getOrDefault(self.num_workers)
        if query_plan[start:start + len("Repartition")] == "Repartition" and \
                num_workers == num_partitions:
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
            num_partitions = dataset.rdd.getNumPartitions()
            query_plan = dataset._sc._jvm.PythonSQLUtils.explainString(
                dataset._jdf.queryExecution(), "extended")
            if self._query_plan_contains_valid_repartition(
                    query_plan, num_partitions):
                return False
        except:  # noqa: E722
            pass
        return True

    def _get_distributed_config(self, dataset, params):
        """
        This just gets the configuration params for distributed xgboost
        """

        classification = self._xgb_cls() == XGBClassifier
        num_classes = int(
            dataset.select(countDistinct('label')).collect()[0][0])
        if classification and num_classes == 2:
            params["objective"] = "binary:logistic"
        elif classification and num_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_classes
        else:
            params["objective"] = "reg:squarederror"

        if self.getOrDefault(self.use_gpu):
            params["tree_method"] = "gpu_hist"
            # TODO: fix this. This only works on databricks runtime.
            #  On open-source spark, we need get the gpu id from the task allocated gpu resources.
            params["gpu_id"] = 0
        params["num_boost_round"] = self.getOrDefault(self.n_estimators)
        xgb_params = self._gen_xgb_params_dict()
        xgb_params.update(params)
        return xgb_params

    @classmethod
    def _get_dist_booster_params(cls, train_params):
        non_booster_params = _get_default_params_from_func(xgboost.train, {})
        booster_params, kwargs_params = {}, {}
        for key, value in train_params.items():
            if key in non_booster_params:
                kwargs_params[key] = value
            else:
                booster_params[key] = value
        return booster_params, kwargs_params

    def _fit_distributed(self, xgb_model_creator, dataset, has_weight,
                         has_validation, fit_params):
        """
        Takes in the dataset, the other parameters, and produces a valid booster
        """
        num_workers = self.getOrDefault(self.num_workers)
        sc = _get_spark_session().sparkContext
        max_concurrent_tasks = _get_max_num_concurrent_tasks(sc)

        if num_workers > max_concurrent_tasks:
            get_logger(self.__class__.__name__) \
                .warning(f'The num_workers {num_workers} set for xgboost distributed '
                         f'training is greater than current max number of concurrent '
                         f'spark task slots, you need wait until more task slots available '
                         f'or you need increase spark cluster workers.')

        if self._repartition_needed(dataset):
            dataset = dataset.withColumn("values", col("values").cast(ArrayType(FloatType())))
            dataset = dataset.repartition(num_workers)
        train_params = self._get_distributed_config(dataset, fit_params)

        def _train_booster(pandas_df_iter):
            """
            Takes in an RDD partition and outputs a booster for that partition after going through
            the Rabit Ring protocol
            """
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()

            use_external_storage = self.getOrDefault(self.use_external_storage)
            external_storage_precision = self.getOrDefault(self.external_storage_precision)
            external_storage_path_prefix = None
            if use_external_storage:
                external_storage_path_prefix = tempfile.mkdtemp()
            dtrain, dval = None, []
            if has_validation:
                dtrain, dval = convert_partition_data_to_dmatrix(
                    pandas_df_iter, has_weight, has_validation,
                    use_external_storage, external_storage_path_prefix, external_storage_precision)
                dval = [(dtrain, "training"), (dval, "validation")]
            else:
                dtrain = convert_partition_data_to_dmatrix(
                    pandas_df_iter, has_weight, has_validation,
                    use_external_storage, external_storage_path_prefix, external_storage_precision)

            booster_params, kwargs_params = self._get_dist_booster_params(
                train_params)
            context.barrier()
            _rabit_args = ""
            if context.partitionId() == 0:
                _rabit_args = str(_get_rabit_args(context, num_workers))

            messages = context.allGather(message=str(_rabit_args))
            _rabit_args = _get_args_from_message_list(messages)
            evals_result = {}
            with RabitContext(_rabit_args, context):
                booster = worker_train(params=booster_params,
                                       dtrain=dtrain,
                                       evals=dval,
                                       evals_result=evals_result,
                                       **kwargs_params)
            context.barrier()

            if use_external_storage:
                shutil.rmtree(external_storage_path_prefix)
            if context.partitionId() == 0:
                yield pd.DataFrame(
                    data={'booster_bytes': [cloudpickle.dumps(booster)]})

        result_ser_booster = dataset.mapInPandas(
            _train_booster,
            schema='booster_bytes binary').rdd.barrier().mapPartitions(
            lambda x: x).collect()[0][0]
        result_xgb_model = self._convert_to_model(
            cloudpickle.loads(result_ser_booster))
        return self._copyValues(self._create_pyspark_model(result_xgb_model))

    def _fit(self, dataset):
        self._validate_params()
        # Unwrap the VectorUDT type column "feature" to 4 primitive columns:
        # ['features.type', 'features.size', 'features.indices', 'features.values']
        features_col = col(self.getOrDefault(self.featuresCol))
        label_col = col(self.getOrDefault(self.labelCol)).alias('label')
        features_array_col = vector_to_array(features_col, dtype="float32").alias("values")
        select_cols = [features_array_col, label_col]

        has_weight = False
        has_validation = False
        has_base_margin = False

        if self.isDefined(self.weightCol) and self.getOrDefault(
                self.weightCol):
            has_weight = True
            select_cols.append(
                col(self.getOrDefault(self.weightCol)).alias('weight'))

        if self.isDefined(self.validationIndicatorCol) and \
                self.getOrDefault(self.validationIndicatorCol):
            has_validation = True
            select_cols.append(
                col(self.getOrDefault(
                    self.validationIndicatorCol)).alias('validationIndicator'))

        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True
            select_cols.append(
                col(self.getOrDefault(self.baseMarginCol)).alias("baseMargin"))

        dataset = dataset.select(*select_cols)
        # create local var `xgb_model_creator` to avoid pickle `self` object to remote worker
        xgb_model_creator = self._get_xgb_model_creator()  # pylint: disable=E1111
        fit_params = self._gen_fit_params_dict()

        if self.getOrDefault(self.num_workers) > 1:
            return self._fit_distributed(xgb_model_creator, dataset, has_weight,
                                         has_validation, fit_params)

        # Note: fit_params will be pickled to remote, it may include `xgb_model` param
        #  which is used as initial model in training. The initial model will be a
        #  `Booster` instance which support pickling.
        def train_func(pandas_df_iter):
            xgb_model = xgb_model_creator()
            train_val_data = prepare_train_val_data(pandas_df_iter, has_weight,
                                                    has_validation,
                                                    has_base_margin)
            # We don't need to handle callbacks param in fit_params specially.
            # User need to ensure callbacks is pickle-able.
            if has_validation:
                train_X, train_y, train_w, train_base_margin, val_X, val_y, val_w, _ = \
                    train_val_data
                eval_set = [(val_X, val_y)]
                sample_weight_eval_set = [val_w]
                # base_margin_eval_set = [val_base_margin] <- the underline
                # Note that on XGBoost 1.2.0, the above doesn't exist.
                xgb_model.fit(train_X,
                              train_y,
                              sample_weight=train_w,
                              base_margin=train_base_margin,
                              eval_set=eval_set,
                              sample_weight_eval_set=sample_weight_eval_set,
                              **fit_params)
            else:
                train_X, train_y, train_w, train_base_margin = train_val_data
                xgb_model.fit(train_X,
                              train_y,
                              sample_weight=train_w,
                              base_margin=train_base_margin,
                              **fit_params)

            ser_model_string = serialize_xgb_model(xgb_model)
            yield pd.DataFrame(data={'model_string': [ser_model_string]})

        # Train on 1 remote worker, return the string of the serialized model
        result_ser_model_string = dataset.repartition(1) \
            .mapInPandas(train_func, schema='model_string string').collect()[0][0]

        # Load model
        result_xgb_model = deserialize_xgb_model(result_ser_model_string,
                                                 xgb_model_creator)
        return self._copyValues(self._create_pyspark_model(result_xgb_model))

    def write(self):
        return XgboostWriter(self)

    @classmethod
    def read(cls):
        return XgboostReader(cls)


class _XgboostModel(Model, _XgboostParams, MLReadable, MLWritable):
    def __init__(self, xgb_sklearn_model=None):
        super().__init__()
        self._xgb_sklearn_model = xgb_sklearn_model

    def get_booster(self):
        """
        Return the `xgboost.core.Booster` instance.
        """
        return self._xgb_sklearn_model.get_booster()

    def get_feature_importances(self, importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Parameters
        ----------
        importance_type: str, default 'weight'
            One of the importance types defined above.
        """
        return self.get_booster().get_score(importance_type=importance_type)

    def write(self):
        return XgboostModelWriter(self)

    @classmethod
    def read(cls):
        return XgboostModelReader(cls)

    def _transform(self, dataset):
        raise NotImplementedError()


class XgboostRegressorModel(_XgboostModel):
    """
    The model returned by :func:`xgboost.spark.XgboostRegressor.fit`

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

        @pandas_udf('double')
        def predict_udf(iterator: Iterator[Tuple[pd.Series]]) \
                -> Iterator[pd.Series]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            X, _, _, _ = prepare_predict_data(iterator, False)
            # Note: In every spark job task, pandas UDF will run in separate python process
            # so it is safe here to call the thread-unsafe model.predict method
            if len(X) > 0:
                preds = xgb_sklearn_model.predict(X, **predict_params)
                yield pd.Series(preds)

        @pandas_udf('double')
        def predict_udf_base_margin(iterator: Iterator[Tuple[pd.Series, pd.Series]]) \
                -> Iterator[pd.Series]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            X, _, _, b_m = prepare_predict_data(iterator, True)
            # Note: In every spark job task, pandas UDF will run in separate python process
            # so it is safe here to call the thread-unsafe model.predict method
            if len(X) > 0:
                preds = xgb_sklearn_model.predict(X,
                                                  base_margin=b_m,
                                                  **predict_params)
                yield pd.Series(preds)

        features_col = col(self.getOrDefault(self.featuresCol))
        features_col = struct(vector_to_array(features_col, dtype="float32").alias("values"))

        has_base_margin = False
        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True

        if has_base_margin:
            base_margin_col = col(self.getOrDefault(self.baseMarginCol))
            pred_col = predict_udf_base_margin(features_col,
                                               base_margin_col)
        else:
            pred_col = predict_udf(features_col)

        predictionColName = self.getOrDefault(self.predictionCol)

        return dataset.withColumn(predictionColName, pred_col)


class XgboostClassifierModel(_XgboostModel, HasProbabilityCol,
                             HasRawPredictionCol):
    """
    The model returned by :func:`xgboost.spark.XgboostClassifier.fit`

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

        @pandas_udf(
            'rawPrediction array<double>, prediction double, probability array<double>'
        )
        def predict_udf(iterator: Iterator[Tuple[pd.Series]]) \
                -> Iterator[pd.DataFrame]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            X, _, _, _ = prepare_predict_data(iterator, False)
            # Note: In every spark job task, pandas UDF will run in separate python process
            # so it is safe here to call the thread-unsafe model.predict method
            if len(X) > 0:
                margins = xgb_sklearn_model.predict(X,
                                                    output_margin=True,
                                                    **predict_params)
                if margins.ndim == 1:
                    # binomial case
                    classone_probs = expit(margins)
                    classzero_probs = 1.0 - classone_probs
                    raw_preds = np.vstack((-margins, margins)).transpose()
                    class_probs = np.vstack(
                        (classzero_probs, classone_probs)).transpose()
                else:
                    # multinomial case
                    raw_preds = margins
                    class_probs = softmax(raw_preds, axis=1)

                # It seems that they use argmax of class probs,
                # not of margin to get the prediction (Note: scala implementation)
                preds = np.argmax(class_probs, axis=1)
                yield pd.DataFrame(
                    data={
                        'rawPrediction': pd.Series(raw_preds.tolist()),
                        'prediction': pd.Series(preds),
                        'probability': pd.Series(class_probs.tolist())
                    })

        @pandas_udf(
            'rawPrediction array<double>, prediction double, probability array<double>'
        )
        def predict_udf_base_margin(iterator: Iterator[Tuple[pd.Series, pd.Series]])\
                -> Iterator[pd.DataFrame]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            X, _, _, b_m = prepare_predict_data(iterator, True)
            # Note: In every spark job task, pandas UDF will run in separate python process
            # so it is safe here to call the thread-unsafe model.predict method
            if len(X) > 0:
                margins = xgb_sklearn_model.predict(X,
                                                    base_margin=b_m,
                                                    output_margin=True,
                                                    **predict_params)
                if margins.ndim == 1:
                    # binomial case
                    classone_probs = expit(margins)
                    classzero_probs = 1.0 - classone_probs
                    raw_preds = np.vstack((-margins, margins)).transpose()
                    class_probs = np.vstack(
                        (classzero_probs, classone_probs)).transpose()
                else:
                    # multinomial case
                    raw_preds = margins
                    class_probs = softmax(raw_preds, axis=1)

                # It seems that they use argmax of class probs,
                # not of margin to get the prediction (Note: scala implementation)
                preds = np.argmax(class_probs, axis=1)
                yield pd.DataFrame(
                    data={
                        'rawPrediction': pd.Series(raw_preds.tolist()),
                        'prediction': pd.Series(preds),
                        'probability': pd.Series(class_probs.tolist())
                    })

        features_col = col(self.getOrDefault(self.featuresCol))
        features_col = struct(vector_to_array(features_col, dtype="float32").alias("values"))

        has_base_margin = False
        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True

        if has_base_margin:
            base_margin_col = col(self.getOrDefault(self.baseMarginCol))
            pred_struct = predict_udf_base_margin(features_col,
                                                  base_margin_col)
        else:
            pred_struct = predict_udf(features_col)

        pred_struct_col = '_prediction_struct'

        rawPredictionColName = self.getOrDefault(self.rawPredictionCol)
        predictionColName = self.getOrDefault(self.predictionCol)
        probabilityColName = self.getOrDefault(self.probabilityCol)
        dataset = dataset.withColumn(pred_struct_col, pred_struct)
        if rawPredictionColName:
            dataset = dataset.withColumn(
                rawPredictionColName,
                array_to_vector(col(pred_struct_col).rawPrediction))
        if predictionColName:
            dataset = dataset.withColumn(predictionColName,
                                         col(pred_struct_col).prediction)
        if probabilityColName:
            dataset = dataset.withColumn(
                probabilityColName,
                array_to_vector(col(pred_struct_col).probability))

        return dataset.drop(pred_struct_col)


def _set_pyspark_xgb_cls_param_attrs(pyspark_estimator_class,
                                     pyspark_model_class):
    params_dict = pyspark_estimator_class._get_xgb_params_default()

    def param_value_converter(v):
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        elif isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        elif isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        else:
            return v

    def set_param_attrs(attr_name, param_obj_):
        param_obj_.typeConverter = param_value_converter
        setattr(pyspark_estimator_class, attr_name, param_obj_)
        setattr(pyspark_model_class, attr_name, param_obj_)

    for name in params_dict.keys():
        if name == 'missing':
            doc = 'Specify the missing value in the features, default np.nan. ' \
                  'We recommend using 0.0 as the missing value for better performance. ' \
                  'Note: In a spark DataFrame, the inactive values in a sparse vector ' \
                  'mean 0 instead of missing values, unless missing=0 is specified.'
        else:
            doc = f'Refer to XGBoost doc of ' \
                  f'{get_class_name(pyspark_estimator_class._xgb_cls())} for this param {name}'

        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    fit_params_dict = pyspark_estimator_class._get_fit_params_default()
    for name in fit_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}' \
              f'.fit() for this param {name}'
        if name == 'callbacks':
            doc += 'The callbacks can be arbitrary functions. It is saved using cloudpickle ' \
                   'which is not a fully self-contained format. It may fail to load with ' \
                   'different versions of dependencies.'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    predict_params_dict = pyspark_estimator_class._get_predict_params_default()
    for name in predict_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}' \
              f'.predict() for this param {name}'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)
