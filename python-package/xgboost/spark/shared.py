#
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name, too-many-ancestors
"""
Shared parameters between pyspark components.
"""

from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasWeightCol


class HasNumClass(Params):
    """
    Mixin for param numClass: the number of classes for classifier.
    """

    numClass = Param(
        Params._dummy(),
        "numClass",
        "number of classes.",
        typeConverter=TypeConverters.toInt,
    )

    def getNumClass(self):
        """
        Gets the value of numClass or its default value.
        """
        return self.getOrDefault(self.numClass)


class _BoosterParams(Params):
    """
    Booster parameters
    """

    eta = Param(
        Params._dummy(),
        "eta",
        "The step size shrinkage used in update to prevents overfitting. "
        "After each boosting step, we can directly get the weights of new features. "
        "and eta actually shrinks the feature weights to make the boosting process "
        "more conservative.",
        typeConverter=TypeConverters.toFloat,
    )

    gamma = Param(
        Params._dummy(),
        "gamma",
        "minimum loss reduction required to make a further partition on a leaf node "
        "of the tree. the larger, the more conservative the algorithm will be.",
        typeConverter=TypeConverters.toFloat,
    )

    maxDepth = Param(
        Params._dummy(),
        "maxDepth",
        "maximum depth of a tree, increase this value will "
        "make model more complex/likely to be overfitting.",
        typeConverter=TypeConverters.toInt,
    )

    maxLeaves = Param(
        Params._dummy(),
        "maxLeaves",
        "Maximum number of nodes to be added. Only relevant when "
        "grow_policy=lossguide is set.",
        typeConverter=TypeConverters.toInt,
    )

    minChildWeight = Param(
        Params._dummy(),
        "minChildWeight",
        "minimum sum of instance weight(hessian) needed in a child. If the "
        "tree partition step results in a leaf node with the sum of instance "
        "weight less than min_child_weight, then the building process will "
        "give up further partitioning. In linear regression mode, this "
        "simply corresponds to minimum number of instances needed to be "
        "in each node. The larger, the more conservative the algorithm "
        "will be.",
        typeConverter=TypeConverters.toFloat,
    )

    maxDeltaStep = Param(
        Params._dummy(),
        "maxDeltaStep",
        "Maximum delta step we allow each tree's weight estimation to be. "
        "If the value is set to 0, it means there is no constraint. If "
        "it is set to a positive value, it can help making the update "
        "step more conservative. Usually this parameter is not needed, "
        "but it might help in logistic regression when class is extremely "
        "imbalanced. Set it to value of 1-10 might help control the update",
        typeConverter=TypeConverters.toFloat,
    )

    subsample = Param(
        Params._dummy(),
        "subsample",
        "subsample ratio of the training instance. Setting it to 0.5 means "
        "that XGBoost randomly collected half of the data instances to grow "
        "trees and this will prevent overfitting.",
        typeConverter=TypeConverters.toFloat,
    )

    colsampleBytree = Param(
        Params._dummy(),
        "colsampleBytree",
        "subsample ratio of columns when constructing each tree.",
        typeConverter=TypeConverters.toFloat,
    )

    colsampleBylevel = Param(
        Params._dummy(),
        "colsampleBylevel",
        "subsample ratio of columns for each split, in each level.",
        typeConverter=TypeConverters.toFloat,
    )

    alpha = Param(
        Params._dummy(),
        "alpha",
        "L1 regularization term on weights, increase this value will make model "
        "more conservative.",
        typeConverter=TypeConverters.toFloat,
    )

    treeMethod = Param(
        Params._dummy(),
        "treeMethod",
        "The tree construction algorithm used in XGBoost. "
        "Options: {'auto', 'exact', 'approx','gpu_hist'} [default='auto']",
        typeConverter=TypeConverters.toString,
    )

    growPolicy = Param(
        Params._dummy(),
        "growPolicy",
        "Controls a way new nodes are added to the tree. Currently supported "
        "only if tree_method is set to hist. Choices: depthwise, lossguide. "
        "depthwise: split at nodes closest to the root. lossguide: split "
        "at nodes with highest loss change.",
        typeConverter=TypeConverters.toString,
    )

    maxBins = Param(
        Params._dummy(),
        "maxBins",
        "maximum number of bins in histogram.",
        typeConverter=TypeConverters.toInt,
    )

    singlePrecisionHistogram = Param(
        Params._dummy(),
        "singlePrecisionHistogram",
        "whether to use single precision to build histograms.",
        typeConverter=TypeConverters.toBoolean,
    )

    sketchEps = Param(
        Params._dummy(),
        "sketchEps",
        "This is only used for approximate greedy algorithm."
        "This roughly translated into O(1 / sketch_eps) number of bins. "
        "Compared to directly select number of bins, this comes with "
        "theoretical guarantee with sketch accuracy, [default=0.03] range: (0, 1)",
        typeConverter=TypeConverters.toFloat,
    )

    scalePosWeight = Param(
        Params._dummy(),
        "scalePosWeight",
        "Control the balance of positive and negative weights, useful for unbalanced classes."
        "A typical value to consider: sum(negative cases) / sum(positive cases)",
        typeConverter=TypeConverters.toFloat,
    )

    sampleType = Param(
        Params._dummy(),
        "sampleType",
        "type of sampling algorithm, options: {'uniform', 'weighted'}",
        typeConverter=TypeConverters.toString,
    )

    normalizeType = Param(
        Params._dummy(),
        "normalizeType",
        "type of normalization algorithm, options: {'tree', 'forest'}",
        typeConverter=TypeConverters.toString,
    )

    rateDrop = Param(
        Params._dummy(),
        "rateDrop",
        "dropout rate",
        typeConverter=TypeConverters.toFloat,
    )

    skipDrop = Param(
        Params._dummy(),
        "skipDrop",
        "probability of skip dropout. If a dropout is skipped, new trees "
        "are added in the same manner as gbtree.",
        typeConverter=TypeConverters.toFloat,
    )

    lambdaBias = Param(
        Params._dummy(),
        "lambdaBias",
        "L2 regularization term on bias, default 0 (no L1 reg on bias "
        "because it is not important) ",
        typeConverter=TypeConverters.toFloat,
    )

    treeLimit = Param(
        Params._dummy(),
        "treeLimit",
        "number of trees used in the prediction; defaults to 0 (use all trees).",
        typeConverter=TypeConverters.toInt,
    )

    monotoneConstraints = Param(
        Params._dummy(),
        "monotoneConstraints",
        "a list in length of number of features, 1 indicate monotonic increasing, "
        "-1 means decreasing, 0 means no constraint. If it is shorter than number "
        "of features, 0 will be padded.",
        typeConverter=TypeConverters.toString,
    )

    interactionConstraints = Param(
        Params._dummy(),
        "interactionConstraints",
        "Constraints for interaction representing permitted interactions. The constraints "
        "must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]], "
        "where each inner list is a group of indices of features that are allowed to "
        "interact with each other. See tutorial for more information",
        typeConverter=TypeConverters.toString,
    )

    def getEta(self):
        """
        Gets the value of eta or its default value.
        """
        return self.getOrDefault(self.eta)

    def getGamma(self):
        """
        Gets the value of gamma or its default value.
        """
        return self.getOrDefault(self.gamma)

    def getMaxDepth(self):
        """
        Gets the value of maxDepth or its default value.
        """
        return self.getOrDefault(self.maxDepth)

    def getMaxLeaves(self):
        """
        Gets the value of maxLeaves or its default value.
        """
        return self.getOrDefault(self.maxLeaves)

    def getMinChildWeight(self):
        """
        Gets the value of minChildWeight or its default value.
        """
        return self.getOrDefault(self.minChildWeight)

    def getMaxDeltaStep(self):
        """
        Gets the value of minChildWeight or its default value.
        """
        return self.getOrDefault(self.maxDeltaStep)

    def getAlpha(self):
        """
        Gets the value of alpha or its default value.
        """
        return self.getOrDefault(self.alpha)

    def getSubsample(self):
        """
        Gets the value of subsample or its default value.
        """
        return self.getOrDefault(self.subsample)

    def getColsampleBytree(self):
        """
        Gets the value of colsampleBytree or its default value.
        """
        return self.getOrDefault(self.colsampleBytree)

    def getColsampleBylevel(self):
        """
        Gets the value of colsampleBylevel or its default value.
        """
        return self.getOrDefault(self.colsampleBylevel)

    def getAlpha(self):
        """
        Gets the value of alpha or its default value.
        """
        return self.getOrDefault(self.alpha)

    def getTreeMethod(self):
        """
        Gets the value of treeMethod or its default value.
        """
        return self.getOrDefault(self.treeMethod)

    def getGrowPolicy(self):
        """
        Gets the value of growPolicy or its default value.
        """
        return self.getOrDefault(self.growPolicy)

    def getMaxBins(self):
        """
        Gets the value of maxBins or its default value.
        """
        return self.getOrDefault(self.maxBins)

    def getSinglePrecisionHistogram(self):
        """
        Gets the value of singlePrecisionHistogram or its default value.
        """
        return self.getOrDefault(self.singlePrecisionHistogram)

    def getSketchEps(self):
        """
        Gets the value of sketchEps or its default value.
        """
        return self.getOrDefault(self.sketchEps)

    def getScalePosWeight(self):
        """
        Gets the value of scalePosWeight or its default value.
        """
        return self.getOrDefault(self.scalePosWeight)

    def getSampleType(self):
        """
        Gets the value of sampleType or its default value.
        """
        return self.getOrDefault(self.sampleType)

    def getNormalizeType(self):
        """
        Gets the value of normalizeType or its default value.
        """
        return self.getOrDefault(self.normalizeType)

    def getRateDrop(self):
        """
        Gets the value of rateDrop or its default value.
        """
        return self.getOrDefault(self.rateDrop)

    def getSkipDrop(self):
        """
        Gets the value of skipDrop or its default value.
        """
        return self.getOrDefault(self.skipDrop)

    def getLambdaBias(self):
        """
        Gets the value of lambdaBias or its default value.
        """
        return self.getOrDefault(self.lambdaBias)

    def getTreeLimit(self):
        """
        Gets the value of treeLimit or its default value.
        """
        return self.getOrDefault(self.treeLimit)

    def getMonotoneConstraints(self):
        """
        Gets the value of monotoneConstraints or its default value.
        """
        return self.getOrDefault(self.monotoneConstraints)

    def getInteractionConstraints(self):
        """
        Gets the value of interactionConstraints or its default value.
        """
        return self.getOrDefault(self.interactionConstraints)


class _LearningTaskParams(Params):
    """Parameters for learning """

    objective = Param(
        Params._dummy(),
        "objective",
        "Specify the learning task and the corresponding learning objective. "
        "options: reg:squarederror, reg:squaredlogerror, reg:logistic, binary:logistic, binary:logitraw"
        "count:poisson, multi:softmax, multi:softprob, rank:pairwise, reg:gamma",
        typeConverter=TypeConverters.toString,
    )

    objectiveType = Param(
        Params._dummy(),
        "objectiveType",
        "The learning objective type of the specified custom objective and eval. "
        "Corresponding type will be assigned if custom objective is defined "
        "options: regression, classification.",
        typeConverter=TypeConverters.toString,
    )

    baseScore = Param(
        Params._dummy(),
        "baseScore",
        "the initial prediction score of all instances, global bias. default=0.5. ",
        typeConverter=TypeConverters.toFloat,
    )

    evalMetric = Param(
        Params._dummy(),
        "evalMetric",
        "evaluation metrics for validation data, a default metric will be assigned "
        "according to objective (rmse for regression, and error for classification, "
        "mean average precision for ranking).",
        typeConverter=TypeConverters.toString,
    )

    trainTestRatio = Param(
        Params._dummy(),
        "trainTestRatio",
        "fraction of training points to use for testing.",
        typeConverter=TypeConverters.toFloat,
    )

    cacheTrainingSet = Param(
        Params._dummy(),
        "cacheTrainingSet",
        "whether caching training data.",
        typeConverter=TypeConverters.toBoolean,
    )

    skipCleanCheckpoint = Param(
        Params._dummy(),
        "skipCleanCheckpoint",
        "whether cleaning checkpoint data",
        typeConverter=TypeConverters.toBoolean,
    )

    numEarlyStoppingRounds = Param(
        Params._dummy(),
        "numEarlyStoppingRounds",
        "number of rounds of decreasing eval metric to tolerate before stopping the training",
        typeConverter=TypeConverters.toInt,
    )

    maximizeEvaluationMetrics = Param(
        Params._dummy(),
        "maximizeEvaluationMetrics",
        "define the expected optimization to the evaluation metrics, true to maximize otherwise "
        "minimize it",
        typeConverter=TypeConverters.toBoolean,
    )

    killSparkContextOnWorkerFailure = Param(
        Params._dummy(),
        "killSparkContextOnWorkerFailure",
        "whether to kill SparkContext when training task fails.",
        typeConverter=TypeConverters.toBoolean,
    )

    def getObjective(self):
        """
        Gets the value of objective or its default value.
        """
        return self.getOrDefault(self.objective)

    def getObjectiveType(self):
        """
        Gets the value of objectiveType or its default value.
        """
        return self.getOrDefault(self.objectiveType)

    def getBaseScore(self):
        """
        Gets the value of baseScore or its default value.
        """
        return self.getOrDefault(self.baseScore)

    def getEvalMetric(self):
        """
        Gets the value of evalMetric or its default value.
        """
        return self.getOrDefault(self.evalMetric)

    def getTrainTestRatio(self):
        """
        Gets the value of trainTestRatio or its default value.
        """
        return self.getOrDefault(self.trainTestRatio)

    def getCacheTrainingSet(self):
        """
        Gets the value of cacheTrainingSet or its default value.
        """
        return self.getOrDefault(self.cacheTrainingSet)

    def getSkipCleanCheckpoint(self):
        """
        Gets the value of skipCleanCheckpoint or its default value.
        """
        return self.getOrDefault(self.skipCleanCheckpoint)

    def getNumEarlyStoppingRounds(self):
        """
        Gets the value of numEarlyStoppingRounds or its default value.
        """
        return self.getOrDefault(self.numEarlyStoppingRounds)

    def getMaximizeEvaluationMetrics(self):
        """
        Gets the value of maximizeEvaluationMetrics or its default value.
        """
        return self.getOrDefault(self.maximizeEvaluationMetrics)

    def getKillSparkContextOnWorkerFailure(self):
        """
        Gets the value of killSparkContextOnWorkerFailure or its default value.
        """
        return self.getOrDefault(self.killSparkContextOnWorkerFailure)


class _GeneralParams(Params):
    """
    The general parameters.
    """

    numRound = Param(
        Params._dummy(),
        "numRound",
        "The number of rounds for boosting.",
        typeConverter=TypeConverters.toInt,
    )

    numWorkers = Param(
        Params._dummy(),
        "numWorkers",
        "The number of workers used to run xgboost.",
        typeConverter=TypeConverters.toInt,
    )

    nthread = Param(
        Params._dummy(),
        "nthread",
        "The number of threads used by per worker.",
        typeConverter=TypeConverters.toInt,
    )

    useExternalMemory = Param(
        Params._dummy(),
        "useExternalMemory",
        "Whether to use external memory as cache.",
        typeConverter=TypeConverters.toBoolean,
    )

    verbosity = Param(
        Params._dummy(),
        "verbosity",
        "Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), "
        "2 (info), 3 (debug).",
        typeConverter=TypeConverters.toInt,
    )

    missing = Param(
        Params._dummy(),
        "missing",
        "The value treated as missing.",
        typeConverter=TypeConverters.toFloat,
    )

    allowNonZeroForMissing = Param(
        Params._dummy(),
        "allowNonZeroForMissing",
        "Allow to have a non-zero value for missing when training or "
        "predicting on a Sparse or Empty vector. Should only be used "
        "if did not use Spark's VectorAssembler class to construct "
        "the feature vector but instead used a method that preserves "
        "zeros in your vector.",
        typeConverter=TypeConverters.toBoolean,
    )

    timeoutRequestWorkers = Param(
        Params._dummy(),
        "timeoutRequestWorkers",
        "the maximum time to request new Workers if numCores are insufficient. "
        "The timeout will be disabled if this value is set smaller than or equal to 0.",
        typeConverter=TypeConverters.toInt,
    )

    checkpointPath = Param(
        Params._dummy(),
        "checkpointPath",
        "the hdfs folder to load and save checkpoints. If there are existing checkpoints "
        "in checkpoint_path. The job will load the checkpoint with highest version as the "
        "starting point for training. If checkpoint_interval is also set, the job will "
        "save a checkpoint every a few rounds.",
        typeConverter=TypeConverters.toString,
    )

    checkpointInterval = Param(
        Params._dummy(),
        "checkpointInterval",
        "set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that "
        "the trained model will get checkpointed every 10 iterations. Note: `checkpoint_path` "
        "must also be set if the checkpoint interval is greater than 0.",
        typeConverter=TypeConverters.toInt,
    )

    seed = Param(
        Params._dummy(), "seed", "Random seed", typeConverter=TypeConverters.toInt
    )

    def getNumRound(self):
        """
        Gets the value of numRound or its default value.
        """
        return self.getOrDefault(self.numRound)

    def getNumWorkers(self):
        """
        Gets the value of numWorkers or its default value.
        """
        return self.getOrDefault(self.numWorkers)

    def getNthread(self):
        """
        Gets the value of nthread or its default value.
        """
        return self.getOrDefault(self.nthread)

    def getUseExternalMemory(self):
        """
        Gets the value of nthread or its default value.
        """
        return self.getOrDefault(self.useExternalMemory)

    def getVerbosity(self):
        """
        Gets the value of verbosity or its default value.
        """
        return self.getOrDefault(self.verbosity)

    def getMissing(self):
        """
        Gets the value of missing or its default value.
        """
        return self.getOrDefault(self.missing)

    def getAllowNonZeroForMissingMissing(self):
        """
        Gets the value of allowNonZeroForMissing or its default value.
        """
        return self.getOrDefault(self.allowNonZeroForMissing)

    def getTimeoutRequestWorkers(self):
        """
        Gets the value of timeoutRequestWorkers or its default value.
        """
        return self.getOrDefault(self.timeoutRequestWorkers)

    def getCheckpointPath(self):
        """
        Gets the value of checkpointPath or its default value.
        """
        return self.getOrDefault(self.checkpointPath)

    def getCheckpointInterval(self):
        """
        Gets the value of checkpointInterval or its default value.
        """
        return self.getOrDefault(self.checkpointInterval)

    def getSeed(self):
        """
        Gets the value of seed or its default value.
        """
        return self.getOrDefault(self.seed)


class HasBaseMarginCol(Params):
    """
    Mixin for param baseMarginCol: baseMargin (aka base margin) column name.
    """

    baseMarginCol = Param(
        Params._dummy(),
        "baseMarginCol",
        "base margin column name.",
        typeConverter=TypeConverters.toString,
    )

    def getBaseMarginCol(self):
        """
        Gets the value of baseMarginCol or its default value.
        """
        return self.getOrDefault(self.baseMarginCol)


class HasLeafPredictionCol(Params):
    """
    Mixin for param leafPredictionCol: leaf prediction column name.
    """

    leafPredictionCol = Param(
        Params._dummy(),
        "leafPredictionCol",
        "leaf prediction column name.",
        typeConverter=TypeConverters.toString,
    )

    def getLeafPredictionCol(self):
        """
        Gets the value of leafPredictionCol or its default value.
        """
        return self.getOrDefault(self.leafPredictionCol)


class HasContribPredictionCol(Params):
    """
    Mixin for param contribPredictionCol: contribution column name.
    """

    contribPredictionCol = Param(
        Params._dummy(),
        "contribPredictionCol",
        "The contribution column name.",
        typeConverter=TypeConverters.toString,
    )

    def getContribPredictionCol(self):
        """
        Gets the value of contribPredictionCol or its default value.
        """
        return self.getOrDefault(self.contribPredictionCol)


class _RabitParams(Params):
    """Rabit parameters passed through Rabit.Init into native layer"""

    rabitRingReduceThreshold = Param(
        Params._dummy(),
        "rabitRingReduceThreshold",
        "threshold count to enable allreduce/broadcast with ring based topology.",
        typeConverter=TypeConverters.toInt,
    )

    rabitTimeout = Param(
        Params._dummy(),
        "rabitTimeout",
        "timeout threshold after rabit observed failures.",
        typeConverter=TypeConverters.toInt,
    )

    rabitConnectRetry = Param(
        Params._dummy(),
        "rabitConnectRetry",
        "number of retry worker do before fail.",
        typeConverter=TypeConverters.toInt,
    )

    def getRabitRingReduceThreshold(self):
        """
        Gets the value of rabitRingReduceThreshold or its default value.
        """
        return self.getOrDefault(self.rabitRingReduceThreshold)

    def getRabitTimeout(self):
        """
        Gets the value of rabitTimeout or its default value.
        """
        return self.getOrDefault(self.rabitTimeout)

    def getRabitConnectRetry(self):
        """
        Gets the value of rabitConnectRetry or its default value.
        """
        return self.getOrDefault(self.rabitConnectRetry)


class _XGBoostCommonParams(
    _GeneralParams,
    _LearningTaskParams,
    _BoosterParams,
    _RabitParams,
    HasWeightCol,
    HasBaseMarginCol,
    HasLeafPredictionCol,
    HasContribPredictionCol,
):
    """
    XGBoost common parameters for both XGBoostClassifier and XGBoostRegressor
    """

    def setNumRound(self, value):
        """
        Sets the value of :py:attr:`numRound`.
        """
        self._set(numRound=value)
        return self

    def setNumWorkers(self, value):
        """
        Sets the value of :py:attr:`numWorkers`.
        """
        self._set(numWorkers=value)
        return self

    def setNthread(self, value):
        """
        Sets the value of :py:attr:`nthread`.
        """
        self._set(nthread=value)
        return self

    def setUseExternalMemory(self, value):
        """
        Sets the value of :py:attr:`useExternalMemory`.
        """
        self._set(useExternalMemory=value)
        return self

    def setVerbosity(self, value):
        """
        Sets the value of :py:attr:`verbosity`.
        """
        self._set(verbosity=value)
        return self

    def setMissing(self, value):
        """
        Sets the value of :py:attr:`missing`.
        """
        self._set(missing=value)
        return self

    def setAllowNonZeroForMissingMissing(self, value):
        """
        Sets the value of :py:attr:`allowNonZeroForMissing`.
        """
        self._set(allowNonZeroForMissing=value)
        return self

    def setTimeoutRequestWorkers(self, value):
        """
        Sets the value of :py:attr:`timeoutRequestWorkers`.
        """
        self._set(timeoutRequestWorkers=value)
        return self

    def setCheckpointPath(self, value):
        """
        Sets the value of :py:attr:`checkpointPath`.
        """
        self._set(checkpointPath=value)
        return self

    def setCheckpointInterval(self, value):
        """
        Sets the value of :py:attr:`checkpointInterval`.
        """
        self._set(checkpointInterval=value)
        return self

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        self._set(seed=value)
        return self

    def setObjective(self, value):
        """
        Sets the value of :py:attr:`objective`.
        """
        return self._set(objective=value)

    def setObjectiveType(self, value):
        """
        Sets the value of :py:attr:`objectiveType`.
        """
        return self._set(objectiveType=value)

    def setBaseScore(self, value):
        """
        Sets the value of :py:attr:`objectiveType`.
        """
        return self._set(baseScore=value)

    def setEvalMetric(self, value):
        """
        Sets the value of :py:attr:`evalMetric`.
        """
        return self._set(evalMetric=value)

    def setTrainTestRatio(self, value):
        """
        Sets the value of :py:attr:`trainTestRatio`.
        """
        return self._set(trainTestRatio=value)

    def setCacheTrainingSet(self, value):
        """
        Sets the value of :py:attr:`cacheTrainingSet`.
        """
        return self._set(cacheTrainingSet=value)

    def setSkipCleanCheckpoint(self, value):
        """
        Sets the value of :py:attr:`skipCleanCheckpoint`.
        """
        return self._set(skipCleanCheckpoint=value)

    def setNumEarlyStoppingRounds(self, value):
        """
        Sets the value of :py:attr:`numEarlyStoppingRounds`.
        """
        return self._set(numEarlyStoppingRounds=value)

    def setMaximizeEvaluationMetrics(self, value):
        """
        Sets the value of :py:attr:`maximizeEvaluationMetrics`.
        """
        return self._set(maximizeEvaluationMetrics=value)

    def setKillSparkContextOnWorkerFailure(self, value):
        """
        Sets the value of :py:attr:`killSparkContextOnWorkerFailure`.
        """
        return self._set(killSparkContextOnWorkerFailure=value)

    def setEta(self, value):
        """
        Sets the value of :py:attr:`eta`.
        """
        return self._set(eta=value)

    def setGamma(self, value):
        """
        Sets the value of :py:attr:`gamma`.
        """
        return self._set(gamma=value)

    def setMaxDepth(self, value):
        """
        Sets the value of :py:attr:`maxDepth`.
        """
        return self._set(maxDepth=value)

    def setMaxLeaves(self, value):
        """
        Sets the value of :py:attr:`maxLeaves`.
        """
        return self._set(maxLeaves=value)

    def setMinChildWeight(self, value):
        """
        Sets the value of :py:attr:`minChildWeight`.
        """
        return self._set(minChildWeight=value)

    def setMaxDeltaStep(self, value):
        """
        Sets the value of :py:attr:`maxDeltaStep`.
        """
        return self._set(maxDeltaStep=value)

    def setAlpha(self, value):
        """
        Sets the value of :py:attr:`alpha`.
        """
        return self._set(alpha=value)

    def setSubsample(self, value):
        """
        Sets the value of :py:attr:`subsample`.
        """
        return self._set(subsample=value)

    def setColsampleBytree(self, value):
        """
        Sets the value of :py:attr:`colsampleBytree`.
        """
        return self._set(colsampleBytree=value)

    def setColsampleBylevel(self, value):
        """
        Sets the value of :py:attr:`colsampleBylevel`.
        """
        return self._set(colsampleBylevel=value)

    def setAlpha(self, value):
        """
        Sets the value of :py:attr:`alpha`.
        """
        return self._set(alpha=value)

    def setTreeMethod(self, value):
        """
        Sets the value of :py:attr:`treeMethod`.
        """
        return self._set(treeMethod=value)

    def setGrowPolicy(self, value):
        """
        Sets the value of :py:attr:`growPolicy`.
        """
        return self._set(growPolicy=value)

    def setMaxBins(self, value):
        """
        Sets the value of :py:attr:`maxBins`.
        """
        return self._set(maxBins=value)

    def setSinglePrecisionHistogram(self, value):
        """
        Sets the value of :py:attr:`singlePrecisionHistogram`.
        """
        return self._set(singlePrecisionHistogram=value)

    def setSketchEps(self, value):
        """
        Sets the value of :py:attr:`sketchEps`.
        """
        return self._set(sketchEps=value)

    def setScalePosWeight(self, value):
        """
        Sets the value of :py:attr:`scalePosWeight`.
        """
        return self._set(scalePosWeight=value)

    def setSampleType(self, value):
        """
        Sets the value of :py:attr:`sampleType`.
        """
        return self._set(sampleType=value)

    def setNormalizeType(self, value):
        """
        Sets the value of :py:attr:`normalizeType`.
        """
        return self._set(normalizeType=value)

    def setRateDrop(self, value):
        """
        Sets the value of :py:attr:`rateDrop`.
        """
        return self._set(rateDrop=value)

    def setSkipDrop(self, value):
        """
        Sets the value of :py:attr:`skipDrop`.
        """
        return self._set(skipDrop=value)

    def setLambdaBias(self, value):
        """
        Sets the value of :py:attr:`lambdaBias`.
        """
        return self._set(lambdaBias=value)

    def setTreeLimit(self, value):
        """
        Sets the value of :py:attr:`treeLimit`.
        """
        return self._set(treeLimit=value)

    def setMonotoneConstraints(self, value):
        """
        Sets the value of :py:attr:`monotoneConstraints`.
        """
        return self._set(monotoneConstraints=value)

    def setInteractionConstraints(self, value):
        """
        Sets the value of :py:attr:`interactionConstraints`.
        """
        return self._set(interactionConstraints=value)


class HasGroupCol(Params):
    """
    Mixin for param groupCol: group column name for regressor.
    """

    groupCol = Param(
        Params._dummy(),
        "groupCol",
        "The group column name.",
        typeConverter=TypeConverters.toString,
    )

    def getGroupCol(self):
        """
        Gets the value of groupCol or its default value.
        """
        return self.getOrDefault(self.groupCol)


class _XGBoostClassifierParams(_XGBoostCommonParams, HasNumClass):
    """
    XGBoostClassifier parameters
    """

    def setNumClass(self, value):
        """
        Sets the value of :py:attr:`numClass`.
        """
        self._set(numClass=value)
        return self


class _XGBoostRegressorParams(_XGBoostCommonParams, HasGroupCol):
    """
    XGBoostRegressor parameters
    """

    def setGroupCol(self, value):
        """
        Sets the value of :py:attr:`numClass`.
        """
        self._set(groupCol=value)
        return self
