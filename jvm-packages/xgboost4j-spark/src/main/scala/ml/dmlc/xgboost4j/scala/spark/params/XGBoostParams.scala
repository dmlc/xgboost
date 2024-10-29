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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.types.{ArrayType, StructType}

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}

trait HasLeafPredictionCol extends Params {
  /**
   * Param for leaf prediction column name.
   *
   * @group param
   */
  final val leafPredictionCol: Param[String] = new Param[String](this, "leafPredictionCol",
    "name of the predictLeaf results")

  /** @group getParam */
  final def getLeafPredictionCol: String = $(leafPredictionCol)
}

trait HasContribPredictionCol extends Params {
  /**
   * Param for contribution prediction column name.
   *
   * @group param
   */
  final val contribPredictionCol: Param[String] = new Param[String](this, "contribPredictionCol",
    "name of the predictContrib results")

  /** @group getParam */
  final def getContribPredictionCol: String = $(contribPredictionCol)
}

trait HasBaseMarginCol extends Params {

  /**
   * Param for initial prediction (aka base margin) column name.
   *
   * @group param
   */
  final val baseMarginCol: Param[String] = new Param[String](this, "baseMarginCol",
    "Initial prediction (aka base margin) column name.")

  /** @group getParam */
  final def getBaseMarginCol: String = $(baseMarginCol)

}

trait HasGroupCol extends Params {

  final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name.")

  /** @group getParam */
  final def getGroupCol: String = $(groupCol)
}

/**
 * Trait for shared param featuresCols.
 */
trait HasFeaturesCols extends Params {
  /**
   * Param for the names of feature columns.
   *
   * @group param
   */
  final val featuresCols: StringArrayParam = new StringArrayParam(this, "featuresCols",
    "An array of feature column names.")

  /** @group getParam */
  final def getFeaturesCols: Array[String] = $(featuresCols)

  /** Check if featuresCols is valid */
  def isFeaturesColsValid: Boolean = {
    isDefined(featuresCols) && $(featuresCols) != Array.empty
  }
}

/**
 * A trait to hold non-xgboost parameters
 */
trait NonXGBoostParams extends Params {
  private val paramNames: ArrayBuffer[String] = ArrayBuffer.empty

  protected def addNonXGBoostParam(ps: Param[_]*): Unit = {
    ps.foreach(p => paramNames.append(p.name))
  }

  protected lazy val nonXGBoostParams: Array[String] = paramNames.toSet.toArray
}

/**
 * XGBoost spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be the XGBoost estimators or models
 */
private[spark] trait SparkParams[T <: Params] extends HasFeaturesCols with HasFeaturesCol
  with HasLabelCol with HasBaseMarginCol with HasWeightCol with HasPredictionCol
  with HasLeafPredictionCol with HasContribPredictionCol
  with RabitParams with NonXGBoostParams with SchemaValidationTrait {

  final val numWorkers = new IntParam(this, "numWorkers", "Number of workers used to train xgboost",
    ParamValidators.gtEq(1))

  final def getNumRound: Int = $(numRound)

  final val forceRepartition = new BooleanParam(this, "forceRepartition", "If the partition " +
    "is equal to numWorkers, xgboost won't repartition the dataset. Set forceRepartition to " +
    "true to force repartition.")

  final def getForceRepartition: Boolean = $(forceRepartition)

  final val numRound = new IntParam(this, "numRound", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  final val numEarlyStoppingRounds = new IntParam(this, "numEarlyStoppingRounds", "Stop training " +
    "Number of rounds of decreasing eval metric to tolerate before stopping training",
    ParamValidators.gtEq(0))

  final def getNumEarlyStoppingRounds: Int = $(numEarlyStoppingRounds)

  final val inferBatchSize = new IntParam(this, "inferBatchSize", "batch size in rows " +
    "to be grouped for inference",
    ParamValidators.gtEq(1))

  /** @group getParam */
  final def getInferBatchSize: Int = $(inferBatchSize)

  /**
   * the value treated as missing. default: Float.NaN
   */
  final val missing = new FloatParam(this, "missing", "The value treated as missing")

  final def getMissing: Float = $(missing)

  final val customObj = new CustomObjParam(this, "customObj", "customized objective function " +
    "provided by user")

  final def getCustomObj: ObjectiveTrait = $(customObj)

  final val customEval = new CustomEvalParam(this, "customEval",
    "customized evaluation function provided by user")

  final def getCustomEval: EvalTrait = $(customEval)

  /** Feature's name, it will be set to DMatrix and Booster, and in the final native json model.
   * In native code, the parameter name is feature_name.
   * */
  final val featureNames = new StringArrayParam(this, "feature_names",
    "an array of feature names")

  final def getFeatureNames: Array[String] = $(featureNames)

  /** Feature types, q is numeric and c is categorical.
   * In native code, the parameter name is feature_type
   * */
  final val featureTypes = new StringArrayParam(this, "feature_types",
    "an array of feature types")

  final def getFeatureTypes: Array[String] = $(featureTypes)

  setDefault(numRound -> 100, numWorkers -> 1, inferBatchSize -> (32 << 10),
    numEarlyStoppingRounds -> 0, forceRepartition -> false, missing -> Float.NaN,
    featuresCols -> Array.empty, customObj -> null, customEval -> null,
    featureNames -> Array.empty, featureTypes -> Array.empty)

  addNonXGBoostParam(numWorkers, numRound, numEarlyStoppingRounds, inferBatchSize, featuresCol,
    labelCol, baseMarginCol, weightCol, predictionCol, leafPredictionCol, contribPredictionCol,
    forceRepartition, featuresCols, customEval, customObj, featureTypes, featureNames)

  final def getNumWorkers: Int = $(numWorkers)

  def setNumWorkers(value: Int): T = set(numWorkers, value).asInstanceOf[T]

  def setForceRepartition(value: Boolean): T = set(forceRepartition, value).asInstanceOf[T]

  def setNumRound(value: Int): T = set(numRound, value).asInstanceOf[T]

  def setFeaturesCol(value: Array[String]): T = set(featuresCols, value).asInstanceOf[T]

  def setBaseMarginCol(value: String): T = set(baseMarginCol, value).asInstanceOf[T]

  def setWeightCol(value: String): T = set(weightCol, value).asInstanceOf[T]

  def setLeafPredictionCol(value: String): T = set(leafPredictionCol, value).asInstanceOf[T]

  def setContribPredictionCol(value: String): T = set(contribPredictionCol, value).asInstanceOf[T]

  def setInferBatchSize(value: Int): T = set(inferBatchSize, value).asInstanceOf[T]

  def setMissing(value: Float): T = set(missing, value).asInstanceOf[T]

  def setCustomObj(value: ObjectiveTrait): T = set(customObj, value).asInstanceOf[T]

  def setCustomEval(value: EvalTrait): T = set(customEval, value).asInstanceOf[T]

  def setRabitTrackerTimeout(value: Int): T = set(rabitTrackerTimeout, value).asInstanceOf[T]

  def setRabitTrackerHostIp(value: String): T = set(rabitTrackerHostIp, value).asInstanceOf[T]

  def setRabitTrackerPort(value: Int): T = set(rabitTrackerPort, value).asInstanceOf[T]

  def setFeatureNames(value: Array[String]): T = set(featureNames, value).asInstanceOf[T]

  def setFeatureTypes(value: Array[String]): T = set(featureTypes, value).asInstanceOf[T]

  protected[spark] def featureIsArrayType(schema: StructType): Boolean =
    schema(getFeaturesCol).dataType.isInstanceOf[ArrayType]

  protected[spark] def validateFeatureType(schema: StructType) = {
    // Features cols must be Vector or Array.
    val featureDataType = schema(getFeaturesCol).dataType

    // Features column must be either ArrayType or VectorType.
    if (!featureDataType.isInstanceOf[ArrayType] && !SparkUtils.isVectorType(featureDataType)) {
      throw new IllegalArgumentException("Feature type must be either ArrayType or VectorType")
    }
  }
}

private[spark] trait SchemaValidationTrait {

  def validateAndTransformSchema(schema: StructType,
                                 fitting: Boolean): StructType = schema
}

/**
 * XGBoost ranking spark-specific parameters
 *
 * @tparam T should be XGBoostRanker or XGBoostRankingModel
 */
private[spark] trait RankerParams[T <: Params] extends HasGroupCol with NonXGBoostParams {
  def setGroupCol(value: String): T = set(groupCol, value).asInstanceOf[T]

  addNonXGBoostParam(groupCol)
}

/**
 * XGBoost-specific parameters to pass into xgboost libraray
 *
 * @tparam T should be the XGBoost estimators or models
 */
private[spark] trait XGBoostParams[T <: Params] extends TreeBoosterParams
  with LearningTaskParams with GeneralParams with DartBoosterParams {

  // Setters for TreeBoosterParams
  def setEta(value: Double): T = set(eta, value).asInstanceOf[T]

  def setGamma(value: Double): T = set(gamma, value).asInstanceOf[T]

  def setMaxDepth(value: Int): T = set(maxDepth, value).asInstanceOf[T]

  def setMinChildWeight(value: Double): T = set(minChildWeight, value).asInstanceOf[T]

  def setMaxDeltaStep(value: Double): T = set(maxDeltaStep, value).asInstanceOf[T]

  def setSubsample(value: Double): T = set(subsample, value).asInstanceOf[T]

  def setSamplingMethod(value: String): T = set(samplingMethod, value).asInstanceOf[T]

  def setColsampleBytree(value: Double): T = set(colsampleBytree, value).asInstanceOf[T]

  def setColsampleBylevel(value: Double): T = set(colsampleBylevel, value).asInstanceOf[T]

  def setColsampleBynode(value: Double): T = set(colsampleBynode, value).asInstanceOf[T]

  def setLambda(value: Double): T = set(lambda, value).asInstanceOf[T]

  def setAlpha(value: Double): T = set(alpha, value).asInstanceOf[T]

  def setTreeMethod(value: String): T = set(treeMethod, value).asInstanceOf[T]

  def setScalePosWeight(value: Double): T = set(scalePosWeight, value).asInstanceOf[T]

  def setUpdater(value: String): T = set(updater, value).asInstanceOf[T]

  def setRefreshLeaf(value: Boolean): T = set(refreshLeaf, value).asInstanceOf[T]

  def setProcessType(value: String): T = set(processType, value).asInstanceOf[T]

  def setGrowPolicy(value: String): T = set(growPolicy, value).asInstanceOf[T]

  def setMaxLeaves(value: Int): T = set(maxLeaves, value).asInstanceOf[T]

  def setMaxBins(value: Int): T = set(maxBins, value).asInstanceOf[T]

  def setNumParallelTree(value: Int): T = set(numParallelTree, value).asInstanceOf[T]

  def setInteractionConstraints(value: String): T =
    set(interactionConstraints, value).asInstanceOf[T]

  def setMaxCachedHistNode(value: Int): T = set(maxCachedHistNode, value).asInstanceOf[T]

  // Setters for LearningTaskParams

  def setObjective(value: String): T = set(objective, value).asInstanceOf[T]

  def setNumClass(value: Int): T = set(numClass, value).asInstanceOf[T]

  def setBaseScore(value: Double): T = set(baseScore, value).asInstanceOf[T]

  def setEvalMetric(value: String): T = set(evalMetric, value).asInstanceOf[T]

  def setSeed(value: Long): T = set(seed, value).asInstanceOf[T]

  def setSeedPerIteration(value: Boolean): T = set(seedPerIteration, value).asInstanceOf[T]

  def setTweedieVariancePower(value: Double): T = set(tweedieVariancePower, value).asInstanceOf[T]

  def setHuberSlope(value: Double): T = set(huberSlope, value).asInstanceOf[T]

  def setAftLossDistribution(value: String): T = set(aftLossDistribution, value).asInstanceOf[T]

  def setLambdarankPairMethod(value: String): T = set(lambdarankPairMethod, value).asInstanceOf[T]

  def setLambdarankNumPairPerSample(value: Int): T =
    set(lambdarankNumPairPerSample, value).asInstanceOf[T]

  def setLambdarankUnbiased(value: Boolean): T = set(lambdarankUnbiased, value).asInstanceOf[T]

  def setLambdarankBiasNorm(value: Double): T = set(lambdarankBiasNorm, value).asInstanceOf[T]

  def setNdcgExpGain(value: Boolean): T = set(ndcgExpGain, value).asInstanceOf[T]

  // Setters for Dart
  def setSampleType(value: String): T = set(sampleType, value).asInstanceOf[T]

  def setNormalizeType(value: String): T = set(normalizeType, value).asInstanceOf[T]

  def setRateDrop(value: Double): T = set(rateDrop, value).asInstanceOf[T]

  def setOneDrop(value: Boolean): T = set(oneDrop, value).asInstanceOf[T]

  def setSkipDrop(value: Double): T = set(skipDrop, value).asInstanceOf[T]

  // Setters for GeneralParams
  def setBooster(value: String): T = set(booster, value).asInstanceOf[T]

  def setDevice(value: String): T = set(device, value).asInstanceOf[T]

  def setVerbosity(value: Int): T = set(verbosity, value).asInstanceOf[T]

  def setValidateParameters(value: Boolean): T = set(validateParameters, value).asInstanceOf[T]

  def setNthread(value: Int): T = set(nthread, value).asInstanceOf[T]
}

private[spark] trait ParamUtils[T <: Params] extends Params {

  def isDefinedNonEmpty(param: Param[String]): Boolean = {
    isDefined(param) && $(param).nonEmpty
  }
}
