from typing import Union, List, Any, Optional, Dict

from pyspark import keyword_only
from pyspark.ml.classification import _JavaProbabilisticClassifier, _JavaProbabilisticClassificationModel

from .params import XGBoostParams


class XGBoostClassifier(_JavaProbabilisticClassifier["XGBoostClassificationModel"], XGBoostParams):
    _input_kwargs: Dict[str, Any]

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        probabilityCol: str = "probability",
        rawPredictionCol: str = "rawPrediction",
        # SparkParams
        numWorkers: Optional[int] = None,
        numRound: Optional[int] = None,
        forceRepartition: Optional[bool] = None,
        numEarlyStoppingRounds: Optional[int] = None,
        inferBatchSize: Optional[int] = None,
        missing: Optional[float] = None,
        useExternalMemory: Optional[bool] = None,
        maxNumDevicePages: Optional[int] = None,
        maxQuantileBatches: Optional[int] = None,
        minCachePageBytes: Optional[int] = None,
        cacheBatchNumber: Optional[int] = None,
        cacheHostRatio: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        # RabitParams
        rabitTrackerTimeout: Optional[int] = None,
        rabitTrackerHostIp: Optional[str] = None,
        rabitTrackerPort: Optional[int] = None,
        # GeneralParams
        booster: Optional[str] = None,
        device: Optional[str] = None,
        verbosity: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        nthread: Optional[int] = None,
        # TreeBoosterParams
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        max_depth: Optional[int] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        sampling_method: Optional[str] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        alpha: Optional[float] = None,
        tree_method: Optional[str] = None,
        scale_pos_weight: Optional[float] = None,
        updater: Optional[str] = None,
        refresh_leaf: Optional[bool] = None,
        process_type: Optional[str] = None,
        grow_policy: Optional[str] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[List[int]] = None,
        interaction_constraints: Optional[str] = None,
        max_cached_hist_node: Optional[int] = None,
        # LearningTaskParams
        objective: Optional[str] = None,
        num_class: Optional[int] = None,
        base_score: Optional[float] = None,
        eval_metric: Optional[str] = None,
        seed: Optional[int] = None,
        seed_per_iteration: Optional[bool] = None,
        tweedie_variance_power: Optional[float] = None,
        huber_slope: Optional[float] = None,
        aft_loss_distribution: Optional[str] = None,
        lambdarank_pair_method: Optional[str] = None,
        lambdarank_num_pair_per_sample: Optional[int] = None,
        lambdarank_unbiased: Optional[bool] = None,
        lambdarank_bias_norm: Optional[float] = None,
        ndcg_exp_gain: Optional[bool] = None,
        # DartBoosterParams
        sample_type: Optional[str] = None,
        normalize_type: Optional[str] = None,
        rate_drop: Optional[float] = None,
        one_drop: Optional[bool] = None,
        skip_drop: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._java_obj = self._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", self.uid
        )
        self._set_params(**self._input_kwargs)

    def _create_model(self, java_model: "JavaObject") -> "XGBoostClassificationModel":
        return XGBoostClassificationModel(java_model)


class XGBoostClassificationModel(_JavaProbabilisticClassificationModel, XGBoostParams):
    pass
