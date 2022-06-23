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

package ml.dmlc.xgboost4j.scala.spark

import scala.collection.{Iterator, mutable}

import ml.dmlc.xgboost4j.scala.spark.params._
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.hadoop.fs.Path

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util._
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.util.{DefaultXGBoostParamsReader, DefaultXGBoostParamsWriter, XGBoostWriter}
import org.apache.spark.sql.types.StructType

class XGBoostRegressor (
    override val uid: String,
    private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostRegressorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("xgbr"), Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(
    Identifiable.randomUID("xgbr"), xgboostParams)

  XGBoost2MLlibParams(xgboostParams)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  def setGroupCol(value: String): this.type = set(groupCol, value)

  // setters for general params
  def setNumRound(value: Int): this.type = set(numRound, value)

  def setNumWorkers(value: Int): this.type = set(numWorkers, value)

  def setNthread(value: Int): this.type = set(nthread, value)

  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  def setSilent(value: Int): this.type = set(silent, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setCheckpointPath(value: String): this.type = set(checkpointPath, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setEta(value: Double): this.type = set(eta, value)

  def setGamma(value: Double): this.type = set(gamma, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  def setSubsample(value: Double): this.type = set(subsample, value)

  def setColsampleBytree(value: Double): this.type = set(colsampleBytree, value)

  def setColsampleBylevel(value: Double): this.type = set(colsampleBylevel, value)

  def setLambda(value: Double): this.type = set(lambda, value)

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  def setGrowPolicy(value: String): this.type = set(growPolicy, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setMaxLeaves(value: Int): this.type = set(maxLeaves, value)

  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)

  def setSampleType(value: String): this.type = set(sampleType, value)

  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  // setters for learning params
  def setObjective(value: String): this.type = set(objective, value)

  def setObjectiveType(value: String): this.type = set(objectiveType, value)

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  def setEvalMetric(value: String): this.type = set(evalMetric, value)

  def setTrainTestRatio(value: Double): this.type = set(trainTestRatio, value)

  def setNumEarlyStoppingRounds(value: Int): this.type = set(numEarlyStoppingRounds, value)

  def setMaximizeEvaluationMetrics(value: Boolean): this.type =
    set(maximizeEvaluationMetrics, value)

  def setCustomObj(value: ObjectiveTrait): this.type = set(customObj, value)

  def setCustomEval(value: EvalTrait): this.type = set(customEval, value)

  def setAllowNonZeroForMissing(value: Boolean): this.type = set(
    allowNonZeroForMissing,
    value
  )

  def setSinglePrecisionHistogram(value: Boolean): this.type =
    set(singlePrecisionHistogram, value)

  // called at the start of fit/train when 'eval_metric' is not defined
  private def setupDefaultEvalMetric(): String = {
    require(isDefined(objective), "Users must set \'objective\' via xgboostParams.")
    if ($(objective).startsWith("rank")) {
      "map"
    } else {
      "rmse"
    }
  }

  private[spark] def transformSchemaInternal(schema: StructType): StructType = {
    if (isFeaturesColSet(schema)) {
      // User has vectorized the features into VectorUDT.
      super.transformSchema(schema)
    } else {
      transformSchemaWithFeaturesCols(false, schema)
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    PreXGBoost.transformSchema(this, schema)
  }

  override protected def train(dataset: Dataset[_]): XGBoostRegressionModel = {

    if (!isDefined(objective)) {
      // If user doesn't set objective, force it to reg:squarederror
      setObjective("reg:squarederror")
    }

    if (!isDefined(evalMetric) || $(evalMetric).isEmpty) {
      set(evalMetric, setupDefaultEvalMetric())
    }

    if (isDefined(customObj) && $(customObj) != null) {
      set(objectiveType, "regression")
    }

    transformSchema(dataset.schema, logging = true)

    // Packing with all params plus params user defined
    val derivedXGBParamMap = xgboostParams ++ MLlib2XGBoostParams
    val buildTrainingData = PreXGBoost.buildDatasetToRDD(this, dataset, derivedXGBParamMap)

    // All non-null param maps in XGBoostRegressor are in derivedXGBParamMap.
    val (_booster, _metrics) = XGBoost.trainDistributed(dataset.sparkSession.sparkContext,
      buildTrainingData, derivedXGBParamMap)

    val model = new XGBoostRegressionModel(uid, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary)
    model
  }

  override def copy(extra: ParamMap): XGBoostRegressor = defaultCopy(extra)
}

object XGBoostRegressor extends DefaultParamsReadable[XGBoostRegressor] {

  override def load(path: String): XGBoostRegressor = super.load(path)
}

class XGBoostRegressionModel private[ml] (
    override val uid: String,
    private[scala] val _booster: Booster)
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with XGBoostRegressorParams with InferenceParams
    with MLWritable with Serializable {

  import XGBoostRegressionModel._

  // only called in copy()
  def this(uid: String) = this(uid, null)

  /**
   * Get the native booster instance of this model.
   * This is used to call low-level APIs on native booster, such as "getFeatureScore".
   */
  def nativeBooster: Booster = _booster

  private var trainingSummary: Option[XGBoostTrainingSummary] = None

  /**
   * Returns summary (e.g. train/test objective history) of model on the
   * training set. An exception is thrown if no summary is available.
   */
  def summary: XGBoostTrainingSummary = trainingSummary.getOrElse {
    throw new IllegalStateException("No training summary available for this XGBoostModel")
  }

  private[spark] def setSummary(summary: XGBoostTrainingSummary): this.type = {
    trainingSummary = Some(summary)
    this
  }

  def setLeafPredictionCol(value: String): this.type = set(leafPredictionCol, value)

  def setContribPredictionCol(value: String): this.type = set(contribPredictionCol, value)

  def setTreeLimit(value: Int): this.type = set(treeLimit, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setAllowNonZeroForMissing(value: Boolean): this.type = set(
    allowNonZeroForMissing,
    value
  )

  def setInferBatchSize(value: Int): this.type = set(inferBatchSize, value)

  /**
   * Single instance prediction.
   * Note: The performance is not ideal, use it carefully!
   */
  override def predict(features: Vector): Double = {
    import ml.dmlc.xgboost4j.scala.spark.util.DataUtils._
    val dm = new DMatrix(processMissingValues(
      Iterator(features.asXGB),
      $(missing),
      $(allowNonZeroForMissing)
    ))
    _booster.predict(data = dm)(0)(0)
  }

  private[scala] def produceResultIterator(
      originalRowItr: Iterator[Row],
      predictionItr: Iterator[Row],
      predLeafItr: Iterator[Row],
      predContribItr: Iterator[Row]): Iterator[Row] = {
    // the following implementation is to be improved
    if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(predictionItr).zip(predLeafItr).zip(predContribItr).
        map { case (((originals: Row, prediction: Row), leaves: Row), contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ leaves.toSeq ++ contribs.toSeq)
        }
    } else if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      (!isDefined(contribPredictionCol) || $(contribPredictionCol).isEmpty)) {
      originalRowItr.zip(predictionItr).zip(predLeafItr).
        map { case ((originals: Row, prediction: Row), leaves: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ leaves.toSeq)
        }
    } else if ((!isDefined(leafPredictionCol) || $(leafPredictionCol).isEmpty) &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(predictionItr).zip(predContribItr).
        map { case ((originals: Row, prediction: Row), contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ contribs.toSeq)
        }
    } else {
      originalRowItr.zip(predictionItr).map {
        case (originals: Row, originalPrediction: Row) =>
          Row.fromSeq(originals.toSeq ++ originalPrediction.toSeq)
      }
    }
  }

  private[scala] def producePredictionItrs(booster: Booster, dm: DMatrix):
      Array[Iterator[Row]] = {
    val originalPredictionItr = {
      booster.predict(dm, outPutMargin = false, $(treeLimit)).map(Row(_)).iterator
    }
    val predLeafItr = {
      if (isDefined(leafPredictionCol)) {
        booster.predictLeaf(dm, $(treeLimit)).
          map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    val predContribItr = {
      if (isDefined(contribPredictionCol)) {
        booster.predictContrib(dm, $(treeLimit)).
          map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    Array(originalPredictionItr, predLeafItr, predContribItr)
  }

  private[spark] def transformSchemaInternal(schema: StructType): StructType = {
    if (isFeaturesColSet(schema)) {
      // User has vectorized the features into VectorUDT.
      super.transformSchema(schema)
    } else {
      transformSchemaWithFeaturesCols(false, schema)
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    PreXGBoost.transformSchema(this, schema)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = PreXGBoost.transformDataset(this, dataset)
    var numColsOutput = 0

    val predictUDF = udf { (originalPrediction: mutable.WrappedArray[Float]) =>
      originalPrediction(0).toDouble
    }

    if ($(predictionCol).nonEmpty) {
      outputData = outputData
        .withColumn($(predictionCol), predictUDF(col(_originalPredictionCol)))
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF.drop(col(_originalPredictionCol))
  }

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    val newModel = copyValues(new XGBoostRegressionModel(uid, _booster), extra)
    newModel.setSummary(summary).setParent(parent)
  }

  override def write: MLWriter =
    new XGBoostRegressionModel.XGBoostRegressionModelWriter(this)
}

object XGBoostRegressionModel extends MLReadable[XGBoostRegressionModel] {

  private[scala] val _originalPredictionCol = "_originalPrediction"

  override def read: MLReader[XGBoostRegressionModel] = new XGBoostRegressionModelReader

  override def load(path: String): XGBoostRegressionModel = super.load(path)

  private[XGBoostRegressionModel]
  class XGBoostRegressionModelWriter(instance: XGBoostRegressionModel) extends XGBoostWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
      // Save model data
      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostRegressionModel")
      val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
      instance._booster.saveModel(outputStream, getModelFormat())
      outputStream.close()
    }
  }

  private class XGBoostRegressionModelReader extends MLReader[XGBoostRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[XGBoostRegressionModel].getName

    override def load(path: String): XGBoostRegressionModel = {
      implicit val sc = super.sparkSession.sparkContext

      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostRegressionModel")
      val dataInStream = internalPath.getFileSystem(sc.hadoopConfiguration).open(internalPath)

      val booster = SXGBoost.loadModel(dataInStream)
      val model = new XGBoostRegressionModel(metadata.uid, booster)
      DefaultXGBoostParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
