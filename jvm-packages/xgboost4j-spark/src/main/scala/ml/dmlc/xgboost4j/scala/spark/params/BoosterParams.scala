/*
 Copyright (c) 2014-2022 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark.params

import scala.collection.immutable.HashSet

import org.apache.spark.ml.param.{DoubleParam, IntParam, BooleanParam, Param, Params}

private[spark] trait BoosterParams extends Params {

  /**
   * step size shrinkage used in update to prevents overfitting. After each boosting step, we
   * can directly get the weights of new features and eta actually shrinks the feature weights
   * to make the boosting process more conservative. [default=0.3] range: [0,1]
   */
  final val eta = new DoubleParam(this, "eta", "step size shrinkage used in update to prevents" +
    " overfitting. After each boosting step, we can directly get the weights of new features." +
    " and eta actually shrinks the feature weights to make the boosting process more conservative.",
    (value: Double) => value >= 0 && value <= 1)

  final def getEta: Double = $(eta)

  /**
   * minimum loss reduction required to make a further partition on a leaf node of the tree.
   * the larger, the more conservative the algorithm will be. [default=0] range: [0,
   * Double.MaxValue]
   */
  final val gamma = new DoubleParam(this, "gamma", "minimum loss reduction required to make a " +
    "further partition on a leaf node of the tree. the larger, the more conservative the " +
    "algorithm will be.", (value: Double) => value >= 0)

  final def getGamma: Double = $(gamma)

  /**
   * maximum depth of a tree, increase this value will make model more complex / likely to be
   * overfitting. [default=6] range: [1, Int.MaxValue]
   */
  final val maxDepth = new IntParam(this, "maxDepth", "maximum depth of a tree, increase this " +
    "value will make model more complex/likely to be overfitting.", (value: Int) => value >= 0)

  final def getMaxDepth: Int = $(maxDepth)


  /**
   * Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
   */
  final val maxLeaves = new IntParam(this, "maxLeaves",
    "Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.",
    (value: Int) => value >= 0)

  final def getMaxLeaves: Int = $(maxLeaves)


  /**
   * minimum sum of instance weight(hessian) needed in a child. If the tree partition step results
   * in a leaf node with the sum of instance weight less than min_child_weight, then the building
   * process will give up further partitioning. In linear regression mode, this simply corresponds
   * to minimum number of instances needed to be in each node. The larger, the more conservative
   * the algorithm will be. [default=1] range: [0, Double.MaxValue]
   */
  final val minChildWeight = new DoubleParam(this, "minChildWeight", "minimum sum of instance" +
    " weight(hessian) needed in a child. If the tree partition step results in a leaf node with" +
    " the sum of instance weight less than min_child_weight, then the building process will" +
    " give up further partitioning. In linear regression mode, this simply corresponds to minimum" +
    " number of instances needed to be in each node. The larger, the more conservative" +
    " the algorithm will be.", (value: Double) => value >= 0)

  final def getMinChildWeight: Double = $(minChildWeight)

  /**
   * Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it
   * means there is no constraint. If it is set to a positive value, it can help making the update
   * step more conservative. Usually this parameter is not needed, but it might help in logistic
   * regression when class is extremely imbalanced. Set it to value of 1-10 might help control the
   * update. [default=0] range: [0, Double.MaxValue]
   */
  final val maxDeltaStep = new DoubleParam(this, "maxDeltaStep", "Maximum delta step we allow " +
    "each tree's weight" +
    " estimation to be. If the value is set to 0, it means there is no constraint. If it is set" +
    " to a positive value, it can help making the update step more conservative. Usually this" +
    " parameter is not needed, but it might help in logistic regression when class is extremely" +
    " imbalanced. Set it to value of 1-10 might help control the update",
    (value: Double) => value >= 0)

  final def getMaxDeltaStep: Double = $(maxDeltaStep)

  /**
   * subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly
   * collected half of the data instances to grow trees and this will prevent overfitting.
   * [default=1] range:(0,1]
   */
  final val subsample = new DoubleParam(this, "subsample", "subsample ratio of the training " +
    "instance. Setting it to 0.5 means that XGBoost randomly collected half of the data " +
    "instances to grow trees and this will prevent overfitting.",
    (value: Double) => value <= 1 && value > 0)

  final def getSubsample: Double = $(subsample)

  /**
   * subsample ratio of columns when constructing each tree. [default=1] range: (0,1]
   */
  final val colsampleBytree = new DoubleParam(this, "colsampleBytree", "subsample ratio of " +
    "columns when constructing each tree.", (value: Double) => value <= 1 && value > 0)

  final def getColsampleBytree: Double = $(colsampleBytree)

  /**
   * subsample ratio of columns for each split, in each level. [default=1] range: (0,1]
   */
  final val colsampleBylevel = new DoubleParam(this, "colsampleBylevel", "subsample ratio of " +
    "columns for each split, in each level.", (value: Double) => value <= 1 && value > 0)

  final def getColsampleBylevel: Double = $(colsampleBylevel)

  /**
   * L2 regularization term on weights, increase this value will make model more conservative.
   * [default=1]
   */
  final val lambda = new DoubleParam(this, "lambda", "L2 regularization term on weights, " +
    "increase this value will make model more conservative.", (value: Double) => value >= 0)

  final def getLambda: Double = $(lambda)

  /**
   * L1 regularization term on weights, increase this value will make model more conservative.
   * [default=0]
   */
  final val alpha = new DoubleParam(this, "alpha", "L1 regularization term on weights, increase " +
    "this value will make model more conservative.", (value: Double) => value >= 0)

  final def getAlpha: Double = $(alpha)

  /**
   * The tree construction algorithm used in XGBoost. options:
   * {'auto', 'exact', 'approx','gpu_hist'} [default='auto']
   */
  final val treeMethod = new Param[String](this, "treeMethod",
    "The tree construction algorithm used in XGBoost, options: " +
      "{'auto', 'exact', 'approx', 'hist', 'gpu_hist'}",
    (value: String) => BoosterParams.supportedTreeMethods.contains(value))

  final def getTreeMethod: String = $(treeMethod)

  /**
   * growth policy for fast histogram algorithm
   */
  final val growPolicy = new Param[String](this, "growPolicy",
    "Controls a way new nodes are added to the tree. Currently supported only if" +
      " tree_method is set to hist. Choices: depthwise, lossguide. depthwise: split at nodes" +
      " closest to the root. lossguide: split at nodes with highest loss change.",
    (value: String) => BoosterParams.supportedGrowthPolicies.contains(value))

  final def getGrowPolicy: String = $(growPolicy)

  /**
   * maximum number of bins in histogram
   */
  final val maxBins = new IntParam(this, "maxBin", "maximum number of bins in histogram",
    (value: Int) => value > 0)

  final def getMaxBins: Int = $(maxBins)

  /**
   * whether to build histograms using single precision floating point values
   */
  final val singlePrecisionHistogram = new BooleanParam(this, "singlePrecisionHistogram",
    "whether to use single precision to build histograms")

  final def getSinglePrecisionHistogram: Boolean = $(singlePrecisionHistogram)

  /**
   * Control the balance of positive and negative weights, useful for unbalanced classes. A typical
   * value to consider: sum(negative cases) / sum(positive cases).   [default=1]
   */
  final val scalePosWeight = new DoubleParam(this, "scalePosWeight", "Control the balance of " +
    "positive and negative weights, useful for unbalanced classes. A typical value to consider:" +
    " sum(negative cases) / sum(positive cases)")

  final def getScalePosWeight: Double = $(scalePosWeight)

  // Dart boosters

  /**
   * Parameter for Dart booster.
   * Type of sampling algorithm. "uniform": dropped trees are selected uniformly.
   * "weighted": dropped trees are selected in proportion to weight. [default="uniform"]
   */
  final val sampleType = new Param[String](this, "sampleType", "type of sampling algorithm, " +
    "options: {'uniform', 'weighted'}",
    (value: String) => BoosterParams.supportedSampleType.contains(value))

  final def getSampleType: String = $(sampleType)

  /**
   * Parameter of Dart booster.
   * type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
   */
  final val normalizeType = new Param[String](this, "normalizeType", "type of normalization" +
    " algorithm, options: {'tree', 'forest'}",
    (value: String) => BoosterParams.supportedNormalizeType.contains(value))

  final def getNormalizeType: String = $(normalizeType)

  /**
   * Parameter of Dart booster.
   * dropout rate. [default=0.0] range: [0.0, 1.0]
   */
  final val rateDrop = new DoubleParam(this, "rateDrop", "dropout rate", (value: Double) =>
    value >= 0 && value <= 1)

  final def getRateDrop: Double = $(rateDrop)

  /**
   * Parameter of Dart booster.
   * probability of skip dropout. If a dropout is skipped, new trees are added in the same manner
   * as gbtree. [default=0.0] range: [0.0, 1.0]
   */
  final val skipDrop = new DoubleParam(this, "skipDrop", "probability of skip dropout. If" +
    " a dropout is skipped, new trees are added in the same manner as gbtree.",
    (value: Double) => value >= 0 && value <= 1)

  final def getSkipDrop: Double = $(skipDrop)

  // linear booster
  /**
   * Parameter of linear booster
   * L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)
   */
  final val lambdaBias = new DoubleParam(this, "lambdaBias", "L2 regularization term on bias, " +
    "default 0 (no L1 reg on bias because it is not important)", (value: Double) => value >= 0)

  final def getLambdaBias: Double = $(lambdaBias)

  final val treeLimit = new IntParam(this, name = "treeLimit",
    doc = "number of trees used in the prediction; defaults to 0 (use all trees).")
  setDefault(treeLimit, 0)

  final def getTreeLimit: Int = $(treeLimit)

  final val monotoneConstraints = new Param[String](this, name = "monotoneConstraints",
    doc = "a list in length of number of features, 1 indicate monotonic increasing, - 1 means " +
      "decreasing, 0 means no constraint. If it is shorter than number of features, 0 will be " +
      "padded ")

  final def getMonotoneConstraints: String = $(monotoneConstraints)

  final val interactionConstraints = new Param[String](this,
    name = "interactionConstraints",
    doc = "Constraints for interaction representing permitted interactions. The constraints" +
      " must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]]," +
      " where each inner list is a group of indices of features that are allowed to interact" +
      " with each other. See tutorial for more information")

  final def getInteractionConstraints: String = $(interactionConstraints)

}

private[scala] object BoosterParams {

  val supportedBoosters = HashSet("gbtree", "gblinear", "dart")

  val supportedTreeMethods = HashSet("auto", "exact", "approx", "hist", "gpu_hist")

  val supportedGrowthPolicies = HashSet("depthwise", "lossguide")

  val supportedSampleType = HashSet("uniform", "weighted")

  val supportedNormalizeType = HashSet("tree", "forest")
}
