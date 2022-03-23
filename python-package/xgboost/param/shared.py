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

    def __init__(self) -> None:
        super(HasNumClass, self).__init__()

    def getNumClass(self):
        """
        Gets the value of numClass or its default value.
        """
        return self.getOrDefault(self.numClass)


class _BoosterParams(Params):
    """
    Booster parameters
    """

    objective = Param(
        Params._dummy(),
        "objective",
        "The objective function used for training.",
        typeConverter=TypeConverters.toString,
    )

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

    def __init__(self, *args):
        super(_BoosterParams, self).__init__(*args)

    def getObjective(self):
        """
        Gets the value of objective or its default value.
        """
        return self.getOrDefault(self.objective)

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

    def __init__(self, *args):
        super(_GeneralParams, self).__init__(*args)

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

    def __init__(self):
        super(HasBaseMarginCol, self).__init__()

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

    def __init__(self):
        super(HasBaseMarginCol, self).__init__()

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

    def __init__(self):
        super(HasContribPredictionCol, self).__init__()

    def getContribPredictionCol(self):
        """
        Gets the value of contribPredictionCol or its default value.
        """
        return self.getOrDefault(self.contribPredictionCol)


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

    def __init__(self):
        super(HasGroupCol, self).__init__()

    def getGroupCol(self):
        """
        Gets the value of groupCol or its default value.
        """
        return self.getOrDefault(self.groupCol)


class _XGBoostCommonParams(
    _BoosterParams,
    _GeneralParams,
    HasLeafPredictionCol,
    HasBaseMarginCol,
    HasContribPredictionCol,
    HasWeightCol,
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

    def setObjective(self, value):
        """
        Sets the value of :py:attr:`objective`.
        """
        self._set(objective=value)
        return self

    def setTreeMethod(self, value):
        """
        Sets the value of :py:attr:`treeMethod`.
        """
        self._set(treeMethod=value)
        return self


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
