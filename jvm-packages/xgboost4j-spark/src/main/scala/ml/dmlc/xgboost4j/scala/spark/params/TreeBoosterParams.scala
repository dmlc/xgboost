/*
 Copyright (c) 2024 by Contributors

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

import org.apache.spark.ml.param._

/**
 * TreeBoosterParams defines the XGBoost TreeBooster parameters for Spark
 *
 * The details can be found at
 * https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
 */
private[spark] trait TreeBoosterParams extends Params {

  final val eta = new DoubleParam(this, "eta", "Step size shrinkage used in update to prevents " +
    "overfitting. After each boosting step, we can directly get the weights of new features, " +
    "and eta shrinks the feature weights to make the boosting process more conservative.",
    ParamValidators.inRange(0, 1, lowerInclusive = true, upperInclusive = true))

  final def getEta: Double = $(eta)

  final val gamma = new DoubleParam(this, "gamma", "Minimum loss reduction required to make a " +
    "further partition on a leaf node of the tree. The larger gamma is, the more conservative " +
    "the algorithm will be.",
    ParamValidators.gtEq(0))

  final def getGamma: Double = $(gamma)

  final val maxDepth = new IntParam(this, "max_depth", "Maximum depth of a tree. Increasing this " +
    "value will make the model more complex and more likely to overfit. 0 indicates no limit " +
    "on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. " +
    "exact tree method requires non-zero value.",
    ParamValidators.gtEq(0))

  final def getMaxDepth: Int = $(maxDepth)

  final val minChildWeight = new DoubleParam(this, "min_child_weight", "Minimum sum of instance " +
    "weight (hessian) needed in a child. If the tree partition step results in a leaf node " +
    "with the sum of instance weight less than min_child_weight, then the building process " +
    "will give up further partitioning. In linear regression task, this simply corresponds " +
    "to minimum number of instances needed to be in each node. The larger min_child_weight " +
    "is, the more conservative the algorithm will be.",
    ParamValidators.gtEq(0))

  final def getMinChildWeight: Double = $(minChildWeight)

  final val maxDeltaStep = new DoubleParam(this, "max_delta_step", "Maximum delta step we allow " +
    "each leaf output to be. If the value is set to 0, it means there is no constraint. If it " +
    "is set to a positive value, it can help making the update step more conservative. Usually " +
    "this parameter is not needed, but it might help in logistic regression when class is " +
    "extremely imbalanced. Set it to value of 1-10 might help control the update.",
    ParamValidators.gtEq(0))

  final def getMaxDeltaStep: Double = $(maxDeltaStep)

  final val subsample = new DoubleParam(this, "subsample", "Subsample ratio of the training " +
    "instances. Setting it to 0.5 means that XGBoost would randomly sample half of the " +
    "training data prior to growing trees. and this will prevent overfitting. Subsampling " +
    "will occur once in every boosting iteration.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getSubsample: Double = $(subsample)

  final val samplingMethod = new Param[String](this, "sampling_method", "The method to use to " +
    "sample the training instances. The supported sampling methods" +
    "uniform: each training instance has an equal probability of being selected. Typically set " +
    "subsample >= 0.5 for good results.\n" +
    "gradient_based: the selection probability for each training instance is proportional to " +
    "the regularized absolute value of gradients. subsample may be set to as low as 0.1 " +
    "without loss of model accuracy. Note that this sampling method is only supported when " +
    "tree_method is set to hist and the device is cuda; other tree methods only support " +
    "uniform sampling.",
    ParamValidators.inArray(Array("uniform", "gradient_based")))

  final def getSamplingMethod: String = $(samplingMethod)

  final val colsampleBytree = new DoubleParam(this, "colsample_bytree", "Subsample ratio of " +
    "columns when constructing each tree. Subsampling occurs once for every tree constructed.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBytree: Double = $(colsampleBytree)


  final val colsampleBylevel = new DoubleParam(this, "colsample_bylevel", "Subsample ratio of " +
    "columns for each level. Subsampling occurs once for every new depth level reached in a " +
    "tree. Columns are subsampled from the set of columns chosen for the current tree.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBylevel: Double = $(colsampleBylevel)


  final val colsampleBynode = new DoubleParam(this, "colsample_bynode", "Subsample ratio of " +
    "columns for each node (split). Subsampling occurs once every time a new split is " +
    "evaluated. Columns are subsampled from the set of columns chosen for the current level.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBynode: Double = $(colsampleBynode)


  /**
   * L2 regularization term on weights, increase this value will make model more conservative.
   * [default=1]
   */
  final val lambda = new DoubleParam(this, "lambda", "L2 regularization term on weights. " +
    "Increasing this value will make model more conservative.", ParamValidators.gtEq(0))

  final def getLambda: Double = $(lambda)

  final val alpha = new DoubleParam(this, "alpha", "L1 regularization term on weights. " +
    "Increasing this value will make model more conservative.", ParamValidators.gtEq(0))

  final def getAlpha: Double = $(alpha)

  final val treeMethod = new Param[String](this, "tree_method", "The tree construction " +
    "algorithm used in XGBoost, options: {'auto', 'exact', 'approx', 'hist', 'gpu_hist'}",
    ParamValidators.inArray(BoosterParams.supportedTreeMethods.toArray))

  final def getTreeMethod: String = $(treeMethod)

  final val scalePosWeight = new DoubleParam(this, "scale_pos_weight", "Control the balance of " +
    "positive and negative weights, useful for unbalanced classes. A typical value to consider: " +
    "sum(negative instances) / sum(positive instances)")

  final def getScalePosWeight: Double = $(scalePosWeight)

  final val updater = new Param[String](this, "updater", "A comma separated string defining the " +
    "sequence of tree updaters to run, providing a modular way to construct and to modify the " +
    "trees. This is an advanced parameter that is usually set automatically, depending on some " +
    "other parameters. However, it could be also set explicitly by a user. " +
    "The following updaters exist:\n" +
    "grow_colmaker: non-distributed column-based construction of trees.\n" +
    "grow_histmaker: distributed tree construction with row-based data splitting based on " +
    "global proposal of histogram counting.\n" +
    "grow_quantile_histmaker: Grow tree using quantized histogram.\n" +
    "grow_gpu_hist: Enabled when tree_method is set to hist along with device=cuda.\n" +
    "grow_gpu_approx: Enabled when tree_method is set to approx along with device=cuda.\n" +
    "sync: synchronizes trees in all distributed nodes.\n" +
    "refresh: refreshes tree's statistics and or leaf values based on the current data. Note " +
    "that no random subsampling of data rows is performed.\n" +
    "prune: prunes the splits where loss < min_split_loss (or gamma) and nodes that have depth " +
    "greater than max_depth.",
    (value: String) => value.split(",").forall(
      ParamValidators.inArray(BoosterParams.supportedUpdaters.toArray)))

  final def getUpdater: String = $(updater)

  final val refreshLeaf = new BooleanParam(this, "refresh_leaf", "This is a parameter of the " +
    "refresh updater. When this flag is 1, tree leafs as well as tree nodes' stats are updated. " +
    "When it is 0, only node stats are updated.")

  final def getRefreshLeaf: Boolean = $(refreshLeaf)

  // TODO set updater/refreshLeaf defaul value
  final val processType = new Param[String](this, "process_type", "A type of boosting process to " +
    "run. options: {default, update}",
    ParamValidators.inArray(Array("default", "update")))

  final def getProcessType: String = $(processType)

  final val growPolicy = new Param[String](this, "grow_policy", "Controls a way new nodes are " +
    "added to the tree. Currently supported only if tree_method is set to hist or approx. " +
    "Choices: depthwise, lossguide. depthwise: split at nodes closest to the root. " +
    "lossguide: split at nodes with highest loss change.",
    ParamValidators.inArray(Array("depthwise", "lossguide")))

  final def getGrowPolicy: String = $(growPolicy)


  final val maxLeaves = new IntParam(this, "max_leaves", "Maximum number of nodes to be added. " +
    "Not used by exact tree method", ParamValidators.gtEq(0))

  final def getMaxLeaves: Int = $(maxLeaves)

  final val maxBins = new IntParam(this, "max_bin", "Maximum number of discrete bins to bucket " +
    "continuous features. Increasing this number improves the optimality of splits at the cost " +
    "of higher computation time. Only used if tree_method is set to hist or approx.",
    ParamValidators.gt(0))

  final def getMaxBins: Int = $(maxBins)

  final val numParallelTree = new IntParam(this, "num_parallel_tree", "Number of parallel trees " +
    "constructed during each iteration. This option is used to support boosted random forest.",
    ParamValidators.gt(0))

  final def getNumParallelTree: Int = $(numParallelTree)

  final val monotoneConstraints = new IntArrayParam(this, "monotone_constraints", "Constraint of " +
    "variable monotonicity.")

  final def getMonotoneConstraints: Array[Int] = $(monotoneConstraints)

  final val interactionConstraints = new Param[String](this,
    name = "interaction_constraints",
    doc = "Constraints for interaction representing permitted interactions. The constraints" +
      " must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]]," +
      " where each inner list is a group of indices of features that are allowed to interact" +
      " with each other. See tutorial for more information")

  final def getInteractionConstraints: String = $(interactionConstraints)


  final val maxCachedHistNode = new IntParam(this, "max_cached_hist_node", "Maximum number of " +
    "cached nodes for CPU histogram.",
    ParamValidators.gt(0))

  final def getMaxCachedHistNode: Int = $(maxCachedHistNode)

  setDefault(eta -> 0.3, gamma -> 0, maxDepth -> 6, minChildWeight -> 1, maxDeltaStep -> 0,
    subsample -> 1, samplingMethod -> "uniform", colsampleBytree -> 1, colsampleBylevel -> 1,
    colsampleBynode -> 1, lambda -> 1, alpha -> 0, treeMethod -> "auto", scalePosWeight -> 1,
    processType -> "default", growPolicy -> "depthwise", maxLeaves -> 0, maxBins -> 256,
    numParallelTree -> 1, maxCachedHistNode -> 65536)

}

private[spark] object BoosterParams {

  val supportedTreeMethods = HashSet("auto", "exact", "approx", "hist", "gpu_hist")

  val supportedUpdaters = HashSet("grow_colmaker", "grow_histmaker", "grow_quantile_histmaker",
    "grow_gpu_hist", "grow_gpu_approx", "sync", "refresh", "prune")
}
