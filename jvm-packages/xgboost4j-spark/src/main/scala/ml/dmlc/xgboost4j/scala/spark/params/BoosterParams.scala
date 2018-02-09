/*
 Copyright (c) 2014 by Contributors

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

import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, Params}

trait BoosterParams extends Params {

  /**
   * Booster to use, options: {'gbtree', 'gblinear', 'dart'}
   */
  val boosterType = new Param[String](this, "booster",
    s"Booster to use, options: {'gbtree', 'gblinear', 'dart'}",
    (value: String) => BoosterParams.supportedBoosters.contains(value.toLowerCase))

  /**
   * step size shrinkage used in update to prevents overfitting. After each boosting step, we
   * can directly get the weights of new features and eta actually shrinks the feature weights
   * to make the boosting process more conservative. [default=0.3] range: [0,1]
   */
  val eta = new DoubleParam(this, "eta", "step size shrinkage used in update to prevents" +
    " overfitting. After each boosting step, we can directly get the weights of new features." +
    " and eta actually shrinks the feature weights to make the boosting process more conservative.",
    (value: Double) => value >= 0 && value <= 1)

  /**
   * minimum loss reduction required to make a further partition on a leaf node of the tree.
   * the larger, the more conservative the algorithm will be. [default=0] range: [0,
   * Double.MaxValue]
   */
  val gamma = new DoubleParam(this, "gamma", "minimum loss reduction required to make a further" +
      " partition on a leaf node of the tree. the larger, the more conservative the algorithm" +
    " will be.", (value: Double) => value >= 0)

  /**
   * maximum depth of a tree, increase this value will make model more complex / likely to be
   * overfitting. [default=6] range: [1, Int.MaxValue]
   */
  val maxDepth = new IntParam(this, "max_depth", "maximum depth of a tree, increase this value" +
    " will make model more complex/likely to be overfitting.", (value: Int) => value >= 1)

  /**
   * minimum sum of instance weight(hessian) needed in a child. If the tree partition step results
   * in a leaf node with the sum of instance weight less than min_child_weight, then the building
   * process will give up further partitioning. In linear regression mode, this simply corresponds
   * to minimum number of instances needed to be in each node. The larger, the more conservative
   * the algorithm will be. [default=1] range: [0, Double.MaxValue]
   */
  val minChildWeight = new DoubleParam(this, "min_child_weight", "minimum sum of instance" +
    " weight(hessian) needed in a child. If the tree partition step results in a leaf node with" +
    " the sum of instance weight less than min_child_weight, then the building process will" +
    " give up further partitioning. In linear regression mode, this simply corresponds to minimum" +
    " number of instances needed to be in each node. The larger, the more conservative" +
    " the algorithm will be.", (value: Double) => value >= 0)

  /**
   * Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it
   * means there is no constraint. If it is set to a positive value, it can help making the update
   * step more conservative. Usually this parameter is not needed, but it might help in logistic
   * regression when class is extremely imbalanced. Set it to value of 1-10 might help control the
   * update. [default=0] range: [0, Double.MaxValue]
   */
  val maxDeltaStep = new DoubleParam(this, "max_delta_step", "Maximum delta step we allow each" +
    " tree's weight" +
    " estimation to be. If the value is set to 0, it means there is no constraint. If it is set" +
    " to a positive value, it can help making the update step more conservative. Usually this" +
    " parameter is not needed, but it might help in logistic regression when class is extremely" +
    " imbalanced. Set it to value of 1-10 might help control the update",
    (value: Double) => value >= 0)

  /**
   * subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly
   * collected half of the data instances to grow trees and this will prevent overfitting.
   * [default=1] range:(0,1]
   */
  val subSample = new DoubleParam(this, "subsample", "subsample ratio of the training instance." +
    " Setting it to 0.5 means that XGBoost randomly collected half of the data instances to" +
    " grow trees and this will prevent overfitting.", (value: Double) => value <= 1 && value > 0)

  /**
   * subsample ratio of columns when constructing each tree. [default=1] range: (0,1]
   */
  val colSampleByTree = new DoubleParam(this, "colsample_bytree", "subsample ratio of columns" +
    " when constructing each tree.", (value: Double) => value <= 1 && value > 0)

  /**
   * subsample ratio of columns for each split, in each level. [default=1] range: (0,1]
   */
  val colSampleByLevel = new DoubleParam(this, "colsample_bylevel", "subsample ratio of columns" +
    " for each split, in each level.", (value: Double) => value <= 1 && value > 0)

  /**
   * L2 regularization term on weights, increase this value will make model more conservative.
   * [default=1]
   */
  val lambda = new DoubleParam(this, "lambda", "L2 regularization term on weights, increase this" +
    " value will make model more conservative.", (value: Double) => value >= 0)

  /**
   * L1 regularization term on weights, increase this value will make model more conservative.
   * [default=0]
   */
  val alpha = new DoubleParam(this, "alpha", "L1 regularization term on weights, increase this" +
    " value will make model more conservative.", (value: Double) => value >= 0)

  /**
   * The tree construction algorithm used in XGBoost. options: {'auto', 'exact', 'approx'}
   *  [default='auto']
   */
  val treeMethod = new Param[String](this, "tree_method",
    "The tree construction algorithm used in XGBoost, options: {'auto', 'exact', 'approx', 'hist'}",
    (value: String) => BoosterParams.supportedTreeMethods.contains(value))

  /**
   * growth policy for fast histogram algorithm
   */
  val growthPolicty = new Param[String](this, "grow_policy",
    "growth policy for fast histogram algorithm",
    (value: String) => BoosterParams.supportedGrowthPolicies.contains(value))

  /**
   * maximum number of bins in histogram
   */
  val maxBins = new IntParam(this, "max_bin", "maximum number of bins in histogram",
    (value: Int) => value > 0)

  /**
   * This is only used for approximate greedy algorithm.
   * This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select
   * number of bins, this comes with theoretical guarantee with sketch accuracy.
   * [default=0.03] range: (0, 1)
   */
  val sketchEps = new DoubleParam(this, "sketch_eps",
    "This is only used for approximate greedy algorithm. This roughly translated into" +
      " O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes" +
      " with theoretical guarantee with sketch accuracy.",
    (value: Double) => value < 1 && value > 0)

  /**
   * Control the balance of positive and negative weights, useful for unbalanced classes. A typical
   * value to consider: sum(negative cases) / sum(positive cases).   [default=0]
   */
  val scalePosWeight = new DoubleParam(this, "scale_pos_weight", "Control the balance of positive" +
    " and negative weights, useful for unbalanced classes. A typical value to consider:" +
    " sum(negative cases) / sum(positive cases)")

  // Dart boosters

  /**
   * Parameter for Dart booster.
   * Type of sampling algorithm. "uniform": dropped trees are selected uniformly.
   * "weighted": dropped trees are selected in proportion to weight. [default="uniform"]
   */
  val sampleType = new Param[String](this, "sample_type", "type of sampling algorithm, options:" +
    " {'uniform', 'weighted'}",
    (value: String) => BoosterParams.supportedSampleType.contains(value))

  /**
   * Parameter of Dart booster.
   * type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
   */
  val normalizeType = new Param[String](this, "normalize_type", "type of normalization" +
    " algorithm, options: {'tree', 'forest'}",
    (value: String) => BoosterParams.supportedNormalizeType.contains(value))

  /**
   * Parameter of Dart booster.
   * dropout rate. [default=0.0] range: [0.0, 1.0]
   */
  val rateDrop = new DoubleParam(this, "rate_drop", "dropout rate", (value: Double) =>
    value >= 0 && value <= 1)

  /**
   * Parameter of Dart booster.
   * probability of skip dropout. If a dropout is skipped, new trees are added in the same manner
   * as gbtree. [default=0.0] range: [0.0, 1.0]
   */
  val skipDrop = new DoubleParam(this, "skip_drop", "probability of skip dropout. If" +
    " a dropout is skipped, new trees are added in the same manner as gbtree.",
    (value: Double) => value >= 0 && value <= 1)

  // linear booster
  /**
   * Parameter of linear booster
   * L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)
   */
  val lambdaBias = new DoubleParam(this, "lambda_bias", "L2 regularization term on bias, default" +
    " 0 (no L1 reg on bias because it is not important)", (value: Double) => value >= 0)

  setDefault(boosterType -> "gbtree", eta -> 0.3, gamma -> 0, maxDepth -> 6,
    minChildWeight -> 1, maxDeltaStep -> 0,
    growthPolicty -> "depthwise", maxBins -> 16,
    subSample -> 1, colSampleByTree -> 1, colSampleByLevel -> 1,
    lambda -> 1, alpha -> 0, treeMethod -> "auto", sketchEps -> 0.03,
    scalePosWeight -> 1.0, sampleType -> "uniform", normalizeType -> "tree",
    rateDrop -> 0.0, skipDrop -> 0.0, lambdaBias -> 0)

  /**
   * Explains all params of this instance. See `explainParam()`.
   */
  override def explainParams(): String = {
    // TODO: filter some parameters according to the booster type
    val boosterTypeStr = $(boosterType)
    val validParamList = {
      if (boosterTypeStr == "gblinear") {
        // gblinear
        params.filter(param => param.name == "lambda" ||
          param.name == "alpha" || param.name == "lambda_bias")
      } else if (boosterTypeStr != "dart") {
        // gbtree
        params.filter(param => param.name != "sample_type" &&
          param.name != "normalize_type" && param.name != "rate_drop" && param.name != "skip_drop")
      } else {
        // dart
        params.filter(_.name != "lambda_bias")
      }
    }
    explainParam(boosterType) + "\n" ++ validParamList.map(explainParam).mkString("\n")
  }
}

private[spark] object BoosterParams {

  val supportedBoosters = HashSet("gbtree", "gblinear", "dart")

  val supportedTreeMethods = HashSet("auto", "exact", "approx", "hist")

  val supportedGrowthPolicies = HashSet("depthwise", "lossguide")

  val supportedSampleType = HashSet("uniform", "weighted")

  val supportedNormalizeType = HashSet("tree", "forest")
}
