from pyspark.ml.param import Param, Params, TypeConverters
from typing import List, TypeVar, Any

P = TypeVar("P", bound=Params)


class DartBoosterParams(Params):
    """
    Parameters specific to the DART (Dropout Additive Regression Trees) boosting algorithm.
    """

    sampleType = Param(
        Params._dummy(),
        "sample_type",
        "Type of sampling algorithm, options: {'uniform', 'weighted'}",
        typeConverter=TypeConverters.toString
    )

    def getSampleType(self) -> str:
        """Gets the value of sampleType or its default value."""
        return self.getOrDefault(self.sampleType)

    normalizeType = Param(
        Params._dummy(),
        "normalize_type",
        "Type of normalization algorithm, options: {'tree', 'forest'}",
        typeConverter=TypeConverters.toString
    )

    def getNormalizeType(self) -> str:
        """Gets the value of normalizeType or its default value."""
        return self.getOrDefault(self.normalizeType)

    rateDrop = Param(
        Params._dummy(),
        "rate_drop",
        "Dropout rate (a fraction of previous trees to drop during the dropout)",
        typeConverter=TypeConverters.toFloat
    )

    def getRateDrop(self) -> float:
        """Gets the value of rateDrop or its default value."""
        return float(self.getOrDefault(self.rateDrop))

    oneDrop = Param(
        Params._dummy(),
        "one_drop",
        "When this flag is enabled, at least one tree is always dropped during the dropout "
        "(allows Binomial-plus-one or epsilon-dropout from the original DART paper)",
        typeConverter=TypeConverters.toBoolean
    )

    def getOneDrop(self) -> bool:
        """Gets the value of oneDrop or its default value."""
        return bool(self.getOrDefault(self.oneDrop))

    skipDrop = Param(
        Params._dummy(),
        "skip_drop",
        "Probability of skipping the dropout procedure during a boosting iteration.\n"
        "If a dropout is skipped, new trees are added in the same manner as gbtree.\n"
        "Note that non-zero skip_drop has higher priority than rate_drop or one_drop.",
        typeConverter=TypeConverters.toFloat
    )

    def getSkipDrop(self) -> float:
        """Gets the value of skipDrop or its default value."""
        return float(self.getOrDefault(self.skipDrop))

    def __init__(self):
        super(DartBoosterParams, self).__init__()
        self._setDefault(
            sampleType="uniform",
            normalizeType="tree",
            rateDrop=0,
            skipDrop=0
        )


class GeneralParams(Params):
    """
    General parameters for XGBoost.
    """

    booster = Param(
        Params._dummy(),
        "booster",
        "Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree "
        "based models while gblinear uses linear functions.",
        typeConverter=TypeConverters.toString
    )

    def getBooster(self) -> str:
        """Gets the value of booster or its default value."""
        return self.getOrDefault(self.booster)

    device = Param(
        Params._dummy(),
        "device",
        "Device for XGBoost to run. User can set it to one of the following values: "
        "{cpu, cuda, gpu}",
        typeConverter=TypeConverters.toString
    )

    def getDevice(self) -> str:
        """Gets the value of device or its default value."""
        return self.getOrDefault(self.device)

    verbosity = Param(
        Params._dummy(),
        "verbosity",
        "Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), "
        "2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based "
        "on heuristics, which is displayed as warning message. If there's unexpected "
        "behaviour, please try to increase value of verbosity.",
        typeConverter=TypeConverters.toInt
    )

    def getVerbosity(self) -> int:
        """Gets the value of verbosity or its default value."""
        return int(self.getOrDefault(self.verbosity))

    validateParameters = Param(
        Params._dummy(),
        "validate_parameters",
        "When set to True, XGBoost will perform validation of input parameters to check "
        "whether a parameter is used or not. A warning is emitted when there's unknown parameter.",
        typeConverter=TypeConverters.toBoolean
    )

    def getValidateParameters(self) -> bool:
        """Gets the value of validateParameters or its default value."""
        return bool(self.getOrDefault(self.validateParameters))

    nthread = Param(
        Params._dummy(),
        "nthread",
        "Number of threads used by per worker",
        typeConverter=TypeConverters.toInt
    )

    def getNthread(self) -> int:
        """Gets the value of nthread or its default value."""
        return int(self.getOrDefault(self.nthread))

    def __init__(self):
        super(GeneralParams, self).__init__()
        self._setDefault(
            booster="gbtree",
            device="cpu",
            verbosity=1,
            validateParameters=False,
            nthread=0
        )


class LearningTaskParams(Params):
    """
    Parameters related to the learning task for XGBoost models.
    """

    objective = Param(
        Params._dummy(),
        "objective",
        "Objective function used for training",
        typeConverter=TypeConverters.toString
    )

    def getObjective(self) -> str:
        """Gets the value of objective or its default value."""
        return self.getOrDefault(self.objective)

    numClass = Param(
        Params._dummy(),
        "num_class",
        "Number of classes, used by multi:softmax and multi:softprob objectives",
        typeConverter=TypeConverters.toInt
    )

    def getNumClass(self) -> int:
        """Gets the value of numClass or its default value."""
        return int(self.getOrDefault(self.numClass))

    baseScore = Param(
        Params._dummy(),
        "base_score",
        "The initial prediction score of all instances, global bias. The parameter is "
        "automatically estimated for selected objectives before training. To disable "
        "the estimation, specify a real number argument. For sufficient number of "
        "iterations, changing this value will not have too much effect.",
        typeConverter=TypeConverters.toFloat
    )

    def getBaseScore(self) -> float:
        """Gets the value of baseScore or its default value."""
        return float(self.getOrDefault(self.baseScore))

    evalMetric = Param(
        Params._dummy(),
        "eval_metric",
        "Evaluation metrics for validation data, a default metric will be assigned "
        "according to objective (rmse for regression, and logloss for classification, "
        "mean average precision for rank:map, etc.) User can add multiple evaluation "
        "metrics. Python users: remember to pass the metrics in as list of parameters "
        "pairs instead of map, so that latter eval_metric won't override previous ones",
        typeConverter=TypeConverters.toString
    )

    def getEvalMetric(self) -> str:
        """Gets the value of evalMetric or its default value."""
        return self.getOrDefault(self.evalMetric)

    seed = Param(
        Params._dummy(),
        "seed",
        "Random number seed.",
        typeConverter=TypeConverters.toInt
    )

    def getSeed(self) -> int:
        """Gets the value of seed or its default value."""
        return int(self.getOrDefault(self.seed))

    seedPerIteration = Param(
        Params._dummy(),
        "seed_per_iteration",
        "Seed PRNG determnisticly via iterator number.",
        typeConverter=TypeConverters.toBoolean
    )

    def getSeedPerIteration(self) -> bool:
        """Gets the value of seedPerIteration or its default value."""
        return bool(self.getOrDefault(self.seedPerIteration))

    tweedieVariancePower = Param(
        Params._dummy(),
        "tweedie_variance_power",
        "Parameter that controls the variance of the Tweedie distribution "
        "var(y) ~ E(y)^tweedie_variance_power.",
        typeConverter=TypeConverters.toFloat
    )

    def getTweedieVariancePower(self) -> float:
        """Gets the value of tweedieVariancePower or its default value."""
        return float(self.getOrDefault(self.tweedieVariancePower))

    huberSlope = Param(
        Params._dummy(),
        "huber_slope",
        "A parameter used for Pseudo-Huber loss to define the (delta) term.",
        typeConverter=TypeConverters.toFloat
    )

    def getHuberSlope(self) -> float:
        """Gets the value of huberSlope or its default value."""
        return float(self.getOrDefault(self.huberSlope))

    aftLossDistribution = Param(
        Params._dummy(),
        "aft_loss_distribution",
        "Probability Density Function",
        typeConverter=TypeConverters.toString
    )

    def getAftLossDistribution(self) -> str:
        """Gets the value of aftLossDistribution or its default value."""
        return self.getOrDefault(self.aftLossDistribution)

    lambdarankPairMethod = Param(
        Params._dummy(),
        "lambdarank_pair_method",
        "pairs for pair-wise learning",
        typeConverter=TypeConverters.toString
    )

    def getLambdarankPairMethod(self) -> str:
        """Gets the value of lambdarankPairMethod or its default value."""
        return self.getOrDefault(self.lambdarankPairMethod)

    lambdarankNumPairPerSample = Param(
        Params._dummy(),
        "lambdarank_num_pair_per_sample",
        "It specifies the number of pairs sampled for each document when pair method is "
        "mean, or the truncation level for queries when the pair method is topk. For "
        "example, to train with ndcg@6, set lambdarank_num_pair_per_sample to 6 and "
        "lambdarank_pair_method to topk",
        typeConverter=TypeConverters.toInt
    )

    def getLambdarankNumPairPerSample(self) -> int:
        """Gets the value of lambdarankNumPairPerSample or its default value."""
        return int(self.getOrDefault(self.lambdarankNumPairPerSample))

    lambdarankUnbiased = Param(
        Params._dummy(),
        "lambdarank_unbiased",
        "Specify whether do we need to debias input click data.",
        typeConverter=TypeConverters.toBoolean
    )

    def getLambdarankUnbiased(self) -> bool:
        """Gets the value of lambdarankUnbiased or its default value."""
        return bool(self.getOrDefault(self.lambdarankUnbiased))

    lambdarankBiasNorm = Param(
        Params._dummy(),
        "lambdarank_bias_norm",
        "Lp normalization for position debiasing, default is L2. Only relevant when "
        "lambdarankUnbiased is set to true.",
        typeConverter=TypeConverters.toFloat
    )

    def getLambdarankBiasNorm(self) -> float:
        """Gets the value of lambdarankBiasNorm or its default value."""
        return float(self.getOrDefault(self.lambdarankBiasNorm))

    ndcgExpGain = Param(
        Params._dummy(),
        "ndcg_exp_gain",
        "Whether we should use exponential gain function for NDCG.",
        typeConverter=TypeConverters.toBoolean
    )

    def getNdcgExpGain(self) -> bool:
        """Gets the value of ndcgExpGain or its default value."""
        return bool(self.getOrDefault(self.ndcgExpGain))

    def __init__(self):
        super(LearningTaskParams, self).__init__()
        self._setDefault(
            objective="reg:squarederror",
            numClass=0,
            seed=0,
            seedPerIteration=False,
            tweedieVariancePower=1.5,
            huberSlope=1,
            lambdarankPairMethod="mean",
            lambdarankUnbiased=False,
            lambdarankBiasNorm=2,
            ndcgExpGain=True
        )


class RabitParams(Params):
    """
    Parameters related to Rabit tracker configuration for distributed XGBoost.
    """

    rabitTrackerTimeout = Param(
        Params._dummy(),
        "rabitTrackerTimeout",
        "The number of seconds before timeout waiting for workers to connect. "
        "and for the tracker to shutdown.",
        typeConverter=TypeConverters.toInt
    )

    def getRabitTrackerTimeout(self) -> int:
        """Gets the value of rabitTrackerTimeout or its default value."""
        return int(self.getOrDefault(self.rabitTrackerTimeout))

    rabitTrackerHostIp = Param(
        Params._dummy(),
        "rabitTrackerHostIp",
        "The Rabit Tracker host IP address. This is only needed if the host IP "
        "cannot be automatically guessed.",
        typeConverter=TypeConverters.toString
    )

    def getRabitTrackerHostIp(self) -> str:
        """Gets the value of rabitTrackerHostIp or its default value."""
        return self.getOrDefault(self.rabitTrackerHostIp)

    rabitTrackerPort = Param(
        Params._dummy(),
        "rabitTrackerPort",
        "The port number for the tracker to listen to. Use a system allocated one by default.",
        typeConverter=TypeConverters.toInt
    )

    def getRabitTrackerPort(self) -> int:
        """Gets the value of rabitTrackerPort or its default value."""
        return int(self.getOrDefault(self.rabitTrackerPort))

    def __init__(self):
        super(RabitParams, self).__init__()
        self._setDefault(
            rabitTrackerTimeout=0,
            rabitTrackerHostIp="",
            rabitTrackerPort=0
        )


class TreeBoosterParams(Params):
    """
    Parameters for Tree Boosting algorithms.
    """

    eta = Param(
        Params._dummy(),
        "eta",
        "Step size shrinkage used in update to prevents overfitting. After each boosting step, "
        "we can directly get the weights of new features, and eta shrinks the feature weights "
        "to make the boosting process more conservative.",
        typeConverter=TypeConverters.toFloat
    )

    def getEta(self) -> float:
        """Gets the value of eta or its default value."""
        return float(self.getOrDefault(self.eta))

    gamma = Param(
        Params._dummy(),
        "gamma",
        "Minimum loss reduction required to make a further partition on a leaf node of the tree. "
        "The larger gamma is, the more conservative the algorithm will be.",
        typeConverter=TypeConverters.toFloat
    )

    def getGamma(self) -> float:
        """Gets the value of gamma or its default value."""
        return float(self.getOrDefault(self.gamma))

    maxDepth = Param(
        Params._dummy(),
        "max_depth",
        "Maximum depth of a tree. Increasing this value will make the model more complex and "
        "more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively "
        "consumes memory when training a deep tree. exact tree method requires non-zero value.",
        typeConverter=TypeConverters.toInt
    )

    def getMaxDepth(self) -> int:
        """Gets the value of maxDepth or its default value."""
        return int(self.getOrDefault(self.maxDepth))

    minChildWeight = Param(
        Params._dummy(),
        "min_child_weight",
        "Minimum sum of instance weight (hessian) needed in a child. If the tree partition "
        "step results in a leaf node with the sum of instance weight less than "
        "min_child_weight, then the building process will give up further partitioning. "
        "In linear regression task, this simply corresponds to minimum number of instances "
        "needed to be in each node. The larger min_child_weight is, the more conservative "
        "the algorithm will be.",
        typeConverter=TypeConverters.toFloat
    )

    def getMinChildWeight(self) -> float:
        """Gets the value of minChildWeight or its default value."""
        return float(self.getOrDefault(self.minChildWeight))

    maxDeltaStep = Param(
        Params._dummy(),
        "max_delta_step",
        "Maximum delta step we allow each leaf output to be. If the value is set to 0, "
        "it means there is no constraint. If it is set to a positive value, it can help "
        "making the update step more conservative. Usually this parameter is not needed, "
        "but it might help in logistic regression when class is extremely imbalanced. "
        "Set it to value of 1-10 might help control the update.",
        typeConverter=TypeConverters.toFloat
    )

    def getMaxDeltaStep(self) -> float:
        """Gets the value of maxDeltaStep or its default value."""
        return float(self.getOrDefault(self.maxDeltaStep))

    subsample = Param(
        Params._dummy(),
        "subsample",
        "Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost "
        "would randomly sample half of the training data prior to growing trees. and this "
        "will prevent overfitting. Subsampling will occur once in every boosting iteration.",
        typeConverter=TypeConverters.toFloat
    )

    def getSubsample(self) -> float:
        """Gets the value of subsample or its default value."""
        return float(self.getOrDefault(self.subsample))

    samplingMethod = Param(
        Params._dummy(),
        "sampling_method",
        "The method to use to sample the training instances. The supported sampling methods "
        "uniform: each training instance has an equal probability of being selected. "
        "Typically set subsample >= 0.5 for good results.\n"
        "gradient_based: the selection probability for each training instance is proportional "
        "to the regularized absolute value of gradients. subsample may be set to as low as "
        "0.1 without loss of model accuracy. Note that this sampling method is only supported "
        "when tree_method is set to hist and the device is cuda; other tree methods only "
        "support uniform sampling.",
        typeConverter=TypeConverters.toString
    )

    def getSamplingMethod(self) -> str:
        """Gets the value of samplingMethod or its default value."""
        return self.getOrDefault(self.samplingMethod)

    colsampleBytree = Param(
        Params._dummy(),
        "colsample_bytree",
        "Subsample ratio of columns when constructing each tree. Subsampling occurs once "
        "for every tree constructed.",
        typeConverter=TypeConverters.toFloat
    )

    def getColsampleBytree(self) -> float:
        """Gets the value of colsampleBytree or its default value."""
        return float(self.getOrDefault(self.colsampleBytree))

    colsampleBylevel = Param(
        Params._dummy(),
        "colsample_bylevel",
        "Subsample ratio of columns for each level. Subsampling occurs once for every new "
        "depth level reached in a tree. Columns are subsampled from the set of columns "
        "chosen for the current tree.",
        typeConverter=TypeConverters.toFloat
    )

    def getColsampleBylevel(self) -> float:
        """Gets the value of colsampleBylevel or its default value."""
        return float(self.getOrDefault(self.colsampleBylevel))

    colsampleBynode = Param(
        Params._dummy(),
        "colsample_bynode",
        "Subsample ratio of columns for each node (split). Subsampling occurs once every "
        "time a new split is evaluated. Columns are subsampled from the set of columns "
        "chosen for the current level.",
        typeConverter=TypeConverters.toFloat
    )

    def getColsampleBynode(self) -> float:
        """Gets the value of colsampleBynode or its default value."""
        return float(self.getOrDefault(self.colsampleBynode))

    # Additional parameters

    lambda_ = Param(
        Params._dummy(),
        "lambda",
        "L2 regularization term on weights. Increasing this value will make model more conservative.",
        typeConverter=TypeConverters.toFloat
    )

    def getLambda(self) -> float:
        """Gets the value of lambda or its default value."""
        return float(self.getOrDefault(self.lambda_))

    alpha = Param(
        Params._dummy(),
        "alpha",
        "L1 regularization term on weights. Increasing this value will make model more conservative.",
        typeConverter=TypeConverters.toFloat
    )

    def getAlpha(self) -> float:
        """Gets the value of alpha or its default value."""
        return float(self.getOrDefault(self.alpha))

    treeMethod = Param(
        Params._dummy(),
        "tree_method",
        "The tree construction algorithm used in XGBoost, options: {'auto', 'exact', 'approx', 'hist', 'gpu_hist'}",
        typeConverter=TypeConverters.toString
    )

    def getTreeMethod(self) -> str:
        """Gets the value of treeMethod or its default value."""
        return self.getOrDefault(self.treeMethod)

    scalePosWeight = Param(
        Params._dummy(),
        "scale_pos_weight",
        "Control the balance of positive and negative weights, useful for unbalanced classes. "
        "A typical value to consider: sum(negative instances) / sum(positive instances)",
        typeConverter=TypeConverters.toFloat
    )

    def getScalePosWeight(self) -> float:
        """Gets the value of scalePosWeight or its default value."""
        return float(self.getOrDefault(self.scalePosWeight))

    updater = Param(
        Params._dummy(),
        "updater",
        "A comma separated string defining the sequence of tree updaters to run, providing a modular "
        "way to construct and to modify the trees. This is an advanced parameter that is usually set "
        "automatically, depending on some other parameters. However, it could be also set explicitly "
        "by a user. The following updaters exist:\n"
        "grow_colmaker: non-distributed column-based construction of trees.\n"
        "grow_histmaker: distributed tree construction with row-based data splitting based on "
        "global proposal of histogram counting.\n"
        "grow_quantile_histmaker: Grow tree using quantized histogram.\n"
        "grow_gpu_hist: Enabled when tree_method is set to hist along with device=cuda.\n"
        "grow_gpu_approx: Enabled when tree_method is set to approx along with device=cuda.\n"
        "sync: synchronizes trees in all distributed nodes.\n"
        "refresh: refreshes tree's statistics and or leaf values based on the current data. Note "
        "that no random subsampling of data rows is performed.\n"
        "prune: prunes the splits where loss < min_split_loss (or gamma) and nodes that have depth "
        "greater than max_depth.",
        typeConverter=TypeConverters.toString
    )

    def getUpdater(self) -> str:
        """Gets the value of updater or its default value."""
        return self.getOrDefault(self.updater)

    refreshLeaf = Param(
        Params._dummy(),
        "refresh_leaf",
        "This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as "
        "tree nodes' stats are updated. When it is 0, only node stats are updated.",
        typeConverter=TypeConverters.toBoolean
    )

    def getRefreshLeaf(self) -> bool:
        """Gets the value of refreshLeaf or its default value."""
        return bool(self.getOrDefault(self.refreshLeaf))

    processType = Param(
        Params._dummy(),
        "process_type",
        "A type of boosting process to run. options: {default, update}",
        typeConverter=TypeConverters.toString
    )

    def getProcessType(self) -> str:
        """Gets the value of processType or its default value."""
        return self.getOrDefault(self.processType)

    growPolicy = Param(
        Params._dummy(),
        "grow_policy",
        "Controls a way new nodes are added to the tree. Currently supported only if tree_method "
        "is set to hist or approx. Choices: depthwise, lossguide. depthwise: split at nodes closest "
        "to the root. lossguide: split at nodes with highest loss change.",
        typeConverter=TypeConverters.toString
    )

    def getGrowPolicy(self) -> str:
        """Gets the value of growPolicy or its default value."""
        return self.getOrDefault(self.growPolicy)

    maxLeaves = Param(
        Params._dummy(),
        "max_leaves",
        "Maximum number of nodes to be added. Not used by exact tree method",
        typeConverter=TypeConverters.toInt
    )

    def getMaxLeaves(self) -> int:
        """Gets the value of maxLeaves or its default value."""
        return int(self.getOrDefault(self.maxLeaves))

    maxBins = Param(
        Params._dummy(),
        "max_bin",
        "Maximum number of discrete bins to bucket continuous features. Increasing this number "
        "improves the optimality of splits at the cost of higher computation time. Only used if "
        "tree_method is set to hist or approx.",
        typeConverter=TypeConverters.toInt
    )

    def getMaxBins(self) -> int:
        """Gets the value of maxBins or its default value."""
        return int(self.getOrDefault(self.maxBins))

    numParallelTree = Param(
        Params._dummy(),
        "num_parallel_tree",
        "Number of parallel trees constructed during each iteration. This option is used to "
        "support boosted random forest.",
        typeConverter=TypeConverters.toInt
    )

    def getNumParallelTree(self) -> int:
        """Gets the value of numParallelTree or its default value."""
        return int(self.getOrDefault(self.numParallelTree))

    monotoneConstraints = Param(
        Params._dummy(),
        "monotone_constraints",
        "Constraint of variable monotonicity.",
        typeConverter=TypeConverters.toListInt
    )

    def getMonotoneConstraints(self) -> List[int]:
        """Gets the value of monotoneConstraints or its default value."""
        return self.getOrDefault(self.monotoneConstraints)

    interactionConstraints = Param(
        Params._dummy(),
        "interaction_constraints",
        "Constraints for interaction representing permitted interactions. The constraints "
        "must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]], "
        "where each inner list is a group of indices of features that are allowed to interact "
        "with each other. See tutorial for more information",
        typeConverter=TypeConverters.toString
    )

    def getInteractionConstraints(self) -> str:
        """Gets the value of interactionConstraints or its default value."""
        return self.getOrDefault(self.interactionConstraints)

    maxCachedHistNode = Param(
        Params._dummy(),
        "max_cached_hist_node",
        "Maximum number of cached nodes for CPU histogram.",
        typeConverter=TypeConverters.toInt
    )

    def getMaxCachedHistNode(self) -> int:
        """Gets the value of maxCachedHistNode or its default value."""
        return int(self.getOrDefault(self.maxCachedHistNode))

    def __init__(self):
        super().__init__()
        self._setDefault(
            eta=0.3, gamma=0, maxDepth=6, minChildWeight=1, maxDeltaStep=0,
            subsample=1, samplingMethod="uniform", colsampleBytree=1, colsampleBylevel=1,
            colsampleBynode=1, lambda_=1, alpha=0, treeMethod="auto", scalePosWeight=1,
            processType="default", growPolicy="depthwise", maxLeaves=0, maxBins=256,
            numParallelTree=1, maxCachedHistNode=65536
        )


class SparkParams(Params):
    """
    Parameters for XGBoost on Spark.
    """

    numWorkers = Param(
        Params._dummy(),
        "numWorkers",
        "Number of workers used to train xgboost",
        typeConverter=TypeConverters.toInt
    )

    def getNumWorkers(self) -> int:
        """Gets the value of numWorkers or its default value."""
        return int(self.getOrDefault(self.numWorkers))

    numRound = Param(
        Params._dummy(),
        "numRound",
        "The number of rounds for boosting",
        typeConverter=TypeConverters.toInt
    )

    def getNumRound(self) -> int:
        """Gets the value of numRound or its default value."""
        return int(self.getOrDefault(self.numRound))

    forceRepartition = Param(
        Params._dummy(),
        "forceRepartition",
        "If the partition is equal to numWorkers, xgboost won't repartition the dataset. "
        "Set forceRepartition to true to force repartition.",
        typeConverter=TypeConverters.toBoolean
    )

    def getForceRepartition(self) -> bool:
        """Gets the value of forceRepartition or its default value."""
        return bool(self.getOrDefault(self.forceRepartition))

    numEarlyStoppingRounds = Param(
        Params._dummy(),
        "numEarlyStoppingRounds",
        "Stop training Number of rounds of decreasing eval metric to tolerate before stopping training",
        typeConverter=TypeConverters.toInt
    )

    def getNumEarlyStoppingRounds(self) -> int:
        """Gets the value of numEarlyStoppingRounds or its default value."""
        return int(self.getOrDefault(self.numEarlyStoppingRounds))

    inferBatchSize = Param(
        Params._dummy(),
        "inferBatchSize",
        "batch size in rows to be grouped for inference",
        typeConverter=TypeConverters.toInt
    )

    def getInferBatchSize(self) -> int:
        """Gets the value of inferBatchSize or its default value."""
        return int(self.getOrDefault(self.inferBatchSize))

    missing = Param(
        Params._dummy(),
        "missing",
        "The value treated as missing",
        typeConverter=TypeConverters.toFloat
    )

    def getMissing(self) -> float:
        """Gets the value of missing or its default value."""
        return float(self.getOrDefault(self.missing))

    featureNames = Param(
        Params._dummy(),
        "feature_names",
        "an array of feature names",
        typeConverter=TypeConverters.toListString
    )

    def getFeatureNames(self) -> List[str]:
        """Gets the value of featureNames or its default value."""
        return self.getOrDefault(self.featureNames)

    featureTypes = Param(
        Params._dummy(),
        "feature_types",
        "an array of feature types",
        typeConverter=TypeConverters.toListString
    )

    def getFeatureTypes(self) -> List[str]:
        """Gets the value of featureTypes or its default value."""
        return self.getOrDefault(self.featureTypes)

    useExternalMemory = Param(
        Params._dummy(),
        "useExternalMemory",
        "Whether to use the external memory or not when building QuantileDMatrix. Please note that "
        "useExternalMemory is useful only when `device` is set to `cuda` or `gpu`. When "
        "useExternalMemory is enabled, the directory specified by spark.local.dir if set will be "
        "used to cache the temporary files, if spark.local.dir is not set, the /tmp directory "
        "will be used.",
        typeConverter=TypeConverters.toBoolean
    )

    def getUseExternalMemory(self) -> bool:
        """Gets the value of useExternalMemory or its default value."""
        return bool(self.getOrDefault(self.useExternalMemory))

    maxNumDevicePages = Param(
        Params._dummy(),
        "maxNumDevicePages",
        "Maximum number of pages cached in device",
        typeConverter=TypeConverters.toInt
    )

    def getMaxNumDevicePages(self) -> int:
        """Gets the value of maxNumDevicePages or its default value."""
        return int(self.getOrDefault(self.maxNumDevicePages))

    maxQuantileBatches = Param(
        Params._dummy(),
        "maxQuantileBatches",
        "Maximum quantile batches",
        typeConverter=TypeConverters.toInt
    )

    def getMaxQuantileBatches(self) -> int:
        """Gets the value of maxQuantileBatches or its default value."""
        return int(self.getOrDefault(self.maxQuantileBatches))

    minCachePageBytes = Param(
        Params._dummy(),
        "minCachePageBytes",
        "Minimum number of bytes for each ellpack page in cache. Only used for in-host",
        typeConverter=TypeConverters.toInt
    )

    def getMinCachePageBytes(self) -> int:
        """Gets the value of minCachePageBytes or its default value."""
        return int(self.getOrDefault(self.minCachePageBytes))

    # Assuming featuresCols is defined elsewhere but referenced in the defaults
    featuresCols = Param(
        Params._dummy(),
        "featuresCols",
        "Feature column names",
        typeConverter=TypeConverters.toListString
    )

    def __init__(self):
        super(SparkParams, self).__init__()
        self._setDefault(
            numRound=100,
            numWorkers=1,
            inferBatchSize=(32 << 10),
            numEarlyStoppingRounds=0,
            forceRepartition=False,
            missing=float("nan"),
            featuresCols=[],
            featureNames=[],
            featureTypes=[],
            useExternalMemory=False,
            maxNumDevicePages=-1,
            maxQuantileBatches=-1,
            minCachePageBytes=-1
        )


class XGBoostParams(SparkParams, DartBoosterParams, GeneralParams,
                    LearningTaskParams, RabitParams, TreeBoosterParams):

    def _set_params(self: "P", **kwargs: Any) -> "P":
        if "featuresCol" in kwargs:
            v = kwargs.pop("featuresCol")
            if isinstance(v, str):
                self._set(**{"featuresCol": v})
            elif isinstance(v, List):
                self._set(**{"featuresCols": v})

        return self._set(**kwargs)
