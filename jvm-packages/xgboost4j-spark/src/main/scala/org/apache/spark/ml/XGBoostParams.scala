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

package org.apache.spark.ml

import org.apache.spark.ml.param._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.types.DoubleType
import scala.collection.mutable.HashMap

object XGBoostParams {

  final val supportedBoosters: Array[String] = Array("gbtree", "gblinear", "dart")

  final val supportedSilentValues: Array[Int] = Array(0, 1)

  final val supportedTreeMethods: Array[String] = Array("auto", "exact", "approx")

  final val supportedSampleType: Array[String] = Array("uniform", "weighted")

  final val supportedNormalizeType: Array[String] = Array("tree", "forest")

  final val supportedObjective: Array[String] = Array("reg:linear", "reg:logistic",
    "binary:logistic", "binary:logitraw", "count:poisson", "multi:softmax",
    "multi:softprob", "rank:pairwise", "reg:gamma")

  final val supportedEvalMetrics: Array[String] = Array("rmse", "mae",
    "logloss", "error", "merror", "mlogloss", "auc", "ndcg", "map",
    "gamma-deviance")
}

/**
 * Abstracts and encapsulates the parameters for XGBoostModel.
 */
trait XGBoostParams extends PredictorParams {

  final val useExternalMemory: BooleanParam = new BooleanParam(this, "use_external_memory",
    s"Flag to indicate whether to use external cache for training and predictions.")

  final val boosterType: Param[String] = new Param[String](this, "booster",
    s"Booster to use ${XGBoostParams.supportedBoosters.mkString(", ")}.",
    (value: String) => XGBoostParams.supportedBoosters.contains(value.toLowerCase))

  final val rounds: IntParam = new IntParam(this, "rounds",
    "Number of iterations/rounds to converge to final predictions.",
    ParamValidators.gtEq(1))

  final val silent: IntParam = new IntParam(this, "silent",
    "0 means printing running messages, 1 means silent mode.",
    (value: Int) => XGBoostParams.supportedSilentValues.contains(value))

  final val eta: DoubleParam = new DoubleParam(this, "eta",
    "Step size shrinkage, used in update to avoid overfiting.",
    ParamValidators.gtEq(0))

  final val gamma: DoubleParam = new DoubleParam(this, "gamma",
    "Min loss reduction required to make a further partition.",
    ParamValidators.gtEq(0))

  final val maxDepth: IntParam = new IntParam(this, "max_depth",
    "Max depth of a tree.",
    ParamValidators.gtEq(1))

  final val numClasses: IntParam = new IntParam(this, "num_class",
    "Number of target classes - needed for multi class classification",
    ParamValidators.gtEq(2))

  final val minChildWeight: DoubleParam = new DoubleParam(this, "min_child_weight",
    "Min sum of instance weight needed in a child.",
    ParamValidators.gtEq(0))

  final val maxDeltaStep: DoubleParam = new DoubleParam(this, "max_delta_step",
    "Max delta step we allow each tree's weight estimation to be.",
    ParamValidators.gtEq(0))

  final val subsample: DoubleParam = new DoubleParam(this, "subsample",
    "Subsample ratio of the training instance.",
    ParamValidators.inRange(0, 1, true, true))

  final val colsampleByTree: DoubleParam = new DoubleParam(this, "colsample_bytree",
    "Subsample ratio of columns when constructing each tree.",
    ParamValidators.inRange(0, 1, true, true))

  final val colsampleByLevel: DoubleParam = new DoubleParam(this, "colsample_bylevel",
    "Subsample ratio of columns for each split, in each level.",
    ParamValidators.inRange(0, 1, true, true))

  final val lambda: DoubleParam = new DoubleParam(this, "lambda",
    "L2 regularization term on weights.",
    ParamValidators.gtEq(0))

  final val alpha: DoubleParam = new DoubleParam(this, "alpha",
    "L1 regularization term on weights.",
    ParamValidators.gtEq(0))

  final val treeMethod: Param[String] = new Param[String](this, "tree_method",
    "The tree constuction algorithm to use.",
    (value: String) => XGBoostParams.supportedTreeMethods.contains(value))

  final val sketchEPS: DoubleParam = new DoubleParam(this, "sketch_eps",
    "Used for greedy algorithm, roughly translates to O(1 / sketchEPS).",
    ParamValidators.inRange(0, 1, true, true))

  final val scalePOSWeight: DoubleParam = new DoubleParam(this, "scale_pos_weight",
    "Control the balance of positive and negative weights.",
    ParamValidators.inRange(0, 1, true, true))

  final val sampleType: Param[String] = new Param[String](this, "sample_type", "Type of sampling"
    + s" algorithm. Must be in ${XGBoostParams.supportedSampleType.mkString(",")}",
    (value: String) => XGBoostParams.supportedSampleType.contains(value))

  final val normalizeType: Param[String] = new Param[String](this, "normalize_type", "Type of" +
    s" normalization algorithm. Must be in" +
    s"${XGBoostParams.supportedNormalizeType.mkString(",")}",
    (value: String) => XGBoostParams.supportedNormalizeType.contains(value))

  final val rateDrop: DoubleParam = new DoubleParam(this, "rate_drop",
    "Dropout rate",
    ParamValidators.inRange(0, 1, true, true))

  final val skipDrop: DoubleParam = new DoubleParam(this, "skip_drop",
    "Probability of skip dropout.",
    ParamValidators.inRange(0, 1, true, true))

  final val lambdaBias: DoubleParam = new DoubleParam(this, "lambda_bias",
    "L2 regularization term on bias, default 0",
    ParamValidators.gtEq(0))

  final val objective: Param[String] = new Param[String](this, "objective",
    "Learning objective.",
    (value: String) => XGBoostParams.supportedObjective.contains(value))

  final val baseScore: DoubleParam = new DoubleParam(this, "base_score",
    "The initial prediction score for all the instances, global bias.",
    ParamValidators.gtEq(0))

  final val evalMetric: Param[String] = new Param[String](this, "eval_metric",
    "Evaluation metric for validation set, chosen automatically based on objective.",
    (value: String) => XGBoostParams.supportedEvalMetrics.contains(value))

  final val seed: IntParam = new IntParam(this, "seed",
    "Random number seed.")

  setDefault(boosterType -> "gbtree", silent -> 0, eta -> 0.3,
    gamma -> 0, maxDepth -> 6, minChildWeight -> 1, maxDeltaStep -> 0,
    subsample -> 1, colsampleByTree -> 1, colsampleByLevel -> 1,
    lambda -> 1, alpha -> 0, treeMethod -> "auto", sketchEPS -> 0.03,
    scalePOSWeight -> 0, sampleType -> "uniform", normalizeType -> "tree",
    rateDrop -> 0.0, skipDrop -> 0.0, lambdaBias -> 0, objective -> "reg:linear",
    baseScore -> 0.5, seed -> 0, numClasses -> 2, rounds -> 10, useExternalMemory -> false)

  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  final def getUseExternalMemory(): Boolean = $(useExternalMemory)

  def setBoosterType(value: String): this.type = set(boosterType, value)

  final def getBoosterType: String = $(boosterType).toLowerCase()

  def setRounds(value: Int): this.type = set(rounds, value)

  final def getRounds: Int = $(rounds)

  def setSilent(value: Int): this.type = set(silent, value)

  final def getSilent: Int = $(silent)

  def setEta(value: Double): this.type = set(eta, value)

  final def getEta: Double = $(eta)

  def setGamma(value: Double): this.type = set(gamma, value)

  final def getGamma: Double = $(gamma)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  final def getMaxDepth: Int = $(maxDepth)

  def setNumClasses(value: Int): this.type = set(numClasses, value)

  final def getNumClasses: Int = $(numClasses)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  def getMinChildWeight: Double = $(minChildWeight)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  final def getMaxDeltaStep: Double = $(maxDeltaStep)

  def setSubsample(value: Double): this.type = set(subsample, value)

  final def getSubsample: Double = $(subsample)

  def setColsampleByTree(value: Double): this.type = set(colsampleByTree, value)

  final def getColsampleByTree: Double = $(colsampleByTree)

  def setColsampleByLevel(value: Double): this.type = set(colsampleByLevel, value)

  final def getColsampleByLevel: Double = $(colsampleByLevel)

  def setLambda(value: Double): this.type = set(lambda, value)

  final def getLambda: Double = $(lambda)

  def setAlpha(value: Double): this.type = set(alpha, value)

  final def getAlpha: Double = $(alpha)

  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  final def getTreeMethod: String = $(treeMethod)

  def setSketchEPS(value: Double): this.type = set(sketchEPS, value)

  final def getSketchEPS: Double = $(sketchEPS)

  def setScalePOSWeight(value: Double): this.type = set(scalePOSWeight, value)

  final def getScalePOSWeight: Double = $(scalePOSWeight)

  def setSampleType(value: String): this.type = set(sampleType, value)

  final def getSampleType: String = $(sampleType)

  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  final def getNormalizeType: String = $(normalizeType)

  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  final def getRateDrop: Double = $(rateDrop)

  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  final def getSkipDrop: Double = $(skipDrop)

  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  final def getLambdaBias: Double = $(lambdaBias)

  def setObjective(value: String): this.type = set(objective, value)

  final def getObjective: String = $(objective)

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  final def getBaseScore: Double = $(baseScore)

  def setSeed(value: Int): this.type = set(seed, value)

  final def getSeed: Int = $(seed)

  def validateAndTransformSchema(schema: StructType, fitting: Boolean): StructType = {

    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT,
      s"Features column ${$(featuresCol)} must be of type Vector.")

    if(fitting) {
        SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType,
          s"Label column ${$(labelCol)} must be of type Double.")
    }

    if (schema.fieldNames.contains($(predictionCol))) {
      throw new IllegalArgumentException(s"Prediction column ${$(predictionCol)} already exists.")
    }

    val outputFields = schema.fields :+ StructField($(predictionCol),
      new VectorUDT, nullable = false)
    StructType(outputFields)
  }

  /**
   * Returns the parameter configuration in the form of Map.
   */
  def paramsMap: Map[String, Any] = {

      val definedParams = new HashMap[String, Any]()
      params.foreach { x => if (isSet(x)) definedParams += (x.name -> $(x)) }
      definedParams.toMap
  }
}
