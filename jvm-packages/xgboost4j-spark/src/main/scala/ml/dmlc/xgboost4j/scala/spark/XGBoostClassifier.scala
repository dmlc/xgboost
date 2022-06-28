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

import ml.dmlc.xgboost4j.scala.spark.params._
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait, ObjectiveTrait, XGBoost => SXGBoost}
import org.apache.hadoop.fs.Path

import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import scala.collection.{Iterator, mutable}

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultXGBoostParamsReader, DefaultXGBoostParamsWriter, XGBoostWriter}
import org.apache.spark.sql.types.StructType

class XGBoostClassifier (
    override val uid: String,
    private[spark] val xgboostParams: Map[String, Any])
  extends ProbabilisticClassifier[Vector, XGBoostClassifier, XGBoostClassificationModel]
    with XGBoostClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("xgbc"), Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(
    Identifiable.randomUID("xgbc"), xgboostParams)

  XGBoost2MLlibParams(xgboostParams)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  def setNumClass(value: Int): this.type = set(numClass, value)

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
    if ($(objective).startsWith("multi")) {
      // multi
      "mlogloss"
    } else {
      // binary
      "logloss"
    }
  }

  // Callback from PreXGBoost
  private[spark] def transformSchemaInternal(schema: StructType): StructType = {
    if (isFeaturesColSet(schema)) {
      // User has vectorized the features into VectorUDT.
      super.transformSchema(schema)
    } else {
      transformSchemaWithFeaturesCols(true, schema)
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    PreXGBoost.transformSchema(this, schema)
  }

  override protected def train(dataset: Dataset[_]): XGBoostClassificationModel = {
    val _numClasses = getNumClasses(dataset)
    if (isDefined(numClass) && $(numClass) != _numClasses) {
      throw new Exception("The number of classes in dataset doesn't match " +
        "\'num_class\' in xgboost params.")
    }

    if (_numClasses == 2) {
      if (!isDefined(objective)) {
        // If user doesn't set objective, force it to binary:logistic
        setObjective("binary:logistic")
      }
    } else if (_numClasses > 2) {
      if (!isDefined(objective)) {
        // If user doesn't set objective, force it to multi:softprob
        setObjective("multi:softprob")
      }
    }

    if (!isDefined(evalMetric) || $(evalMetric).isEmpty) {
      set(evalMetric, setupDefaultEvalMetric())
    }

    if (isDefined(customObj) && $(customObj) != null) {
      set(objectiveType, "classification")
    }

    // Packing with all params plus params user defined
    val derivedXGBParamMap = xgboostParams ++ MLlib2XGBoostParams
    val buildTrainingData = PreXGBoost.buildDatasetToRDD(this, dataset, derivedXGBParamMap)
    transformSchema(dataset.schema, logging = true)

    // All non-null param maps in XGBoostClassifier are in derivedXGBParamMap.
    val (_booster, _metrics) = XGBoost.trainDistributed(dataset.sparkSession.sparkContext,
      buildTrainingData, derivedXGBParamMap)

    val model = new XGBoostClassificationModel(uid, _numClasses, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary)
    model
  }

  override def copy(extra: ParamMap): XGBoostClassifier = defaultCopy(extra)
}

object XGBoostClassifier extends DefaultParamsReadable[XGBoostClassifier] {

  override def load(path: String): XGBoostClassifier = super.load(path)
}

class XGBoostClassificationModel private[ml](
    override val uid: String,
    override val numClasses: Int,
    private[scala] val _booster: Booster)
  extends ProbabilisticClassificationModel[Vector, XGBoostClassificationModel]
    with XGBoostClassifierParams with InferenceParams
    with MLWritable with Serializable {

  import XGBoostClassificationModel._

  // only called in copy()
  def this(uid: String) = this(uid, 2, null)

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
    val probability = _booster.predict(data = dm)(0).map(_.toDouble)
    if (numClasses == 2) {
      math.round(probability(0))
    } else {
      probability2prediction(Vectors.dense(probability))
    }
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override def predictRaw(features: Vector): Vector = {
    throw new Exception("XGBoost-Spark does not support \'predictRaw\'")
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    throw new Exception("XGBoost-Spark does not support \'raw2probabilityInPlace\'")
  }

  private[scala] def produceResultIterator(
      originalRowItr: Iterator[Row],
      rawPredictionItr: Iterator[Row],
      probabilityItr: Iterator[Row],
      predLeafItr: Iterator[Row],
      predContribItr: Iterator[Row]): Iterator[Row] = {
    // the following implementation is to be improved
    if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(rawPredictionItr).zip(probabilityItr).zip(predLeafItr).zip(predContribItr).
        map { case ((((originals: Row, rawPrediction: Row), probability: Row), leaves: Row),
        contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ rawPrediction.toSeq ++ probability.toSeq ++ leaves.toSeq ++
            contribs.toSeq)
      }
    } else if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      (!isDefined(contribPredictionCol) || $(contribPredictionCol).isEmpty)) {
      originalRowItr.zip(rawPredictionItr).zip(probabilityItr).zip(predLeafItr).
        map { case (((originals: Row, rawPrediction: Row), probability: Row), leaves: Row) =>
          Row.fromSeq(originals.toSeq ++ rawPrediction.toSeq ++ probability.toSeq ++ leaves.toSeq)
        }
    } else if ((!isDefined(leafPredictionCol) || $(leafPredictionCol).isEmpty) &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(rawPredictionItr).zip(probabilityItr).zip(predContribItr).
        map { case (((originals: Row, rawPrediction: Row), probability: Row), contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ rawPrediction.toSeq ++ probability.toSeq ++ contribs.toSeq)
        }
    } else {
      originalRowItr.zip(rawPredictionItr).zip(probabilityItr).map {
        case ((originals: Row, rawPrediction: Row), probability: Row) =>
          Row.fromSeq(originals.toSeq ++ rawPrediction.toSeq ++ probability.toSeq)
      }
    }
  }

  private[scala] def producePredictionItrs(booster: Booster, dm: DMatrix):
      Array[Iterator[Row]] = {
    val rawPredictionItr = {
      booster.predict(dm, outPutMargin = true, $(treeLimit)).
        map(Row(_)).iterator
    }
    val probabilityItr = {
      booster.predict(dm, outPutMargin = false, $(treeLimit)).
        map(Row(_)).iterator
    }
    val predLeafItr = {
      if (isDefined(leafPredictionCol)) {
        booster.predictLeaf(dm, $(treeLimit)).map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    val predContribItr = {
      if (isDefined(contribPredictionCol)) {
        booster.predictContrib(dm, $(treeLimit)).map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr)
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
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = PreXGBoost.transformDataset(this, dataset)
    var numColsOutput = 0

    val rawPredictionUDF = udf { rawPrediction: mutable.WrappedArray[Float] =>
      val raw = rawPrediction.map(_.toDouble).toArray
      val rawPredictions = if (numClasses == 2) Array(-raw(0), raw(0)) else raw
      Vectors.dense(rawPredictions)
    }

    if ($(rawPredictionCol).nonEmpty) {
      outputData = outputData
        .withColumn(getRawPredictionCol, rawPredictionUDF(col(_rawPredictionCol)))
      numColsOutput += 1
    }

    if (getObjective.equals("multi:softmax")) {
      // For objective=multi:softmax scenario, there is no probability predicted from xgboost.
      // Instead, the probability column will be filled with real prediction
      val predictUDF = udf { probability: mutable.WrappedArray[Float] =>
        probability(0)
      }
      if ($(predictionCol).nonEmpty) {
        outputData = outputData
          .withColumn($(predictionCol), predictUDF(col(_probabilityCol)))
        numColsOutput += 1
      }

    } else {
      val probabilityUDF = udf { probability: mutable.WrappedArray[Float] =>
        val prob = probability.map(_.toDouble).toArray
        val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
        Vectors.dense(probabilities)
      }
      if ($(probabilityCol).nonEmpty) {
        outputData = outputData
          .withColumn(getProbabilityCol, probabilityUDF(col(_probabilityCol)))
        numColsOutput += 1
      }

      val predictUDF = udf { probability: mutable.WrappedArray[Float] =>
        // From XGBoost probability to MLlib prediction
        val prob = probability.map(_.toDouble).toArray
        val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
        probability2prediction(Vectors.dense(probabilities))
      }
      if ($(predictionCol).nonEmpty) {
        outputData = outputData
          .withColumn($(predictionCol), predictUDF(col(_probabilityCol)))
        numColsOutput += 1
      }
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData
      .toDF
      .drop(col(_rawPredictionCol))
      .drop(col(_probabilityCol))
  }

  override def copy(extra: ParamMap): XGBoostClassificationModel = {
    val newModel = copyValues(new XGBoostClassificationModel(uid, numClasses, _booster), extra)
    newModel.setSummary(summary).setParent(parent)
  }

  override def write: MLWriter =
    new XGBoostClassificationModel.XGBoostClassificationModelWriter(this)
}

object XGBoostClassificationModel extends MLReadable[XGBoostClassificationModel] {

  private[scala] val _rawPredictionCol = "_rawPrediction"
  private[scala] val _probabilityCol = "_probability"

  override def read: MLReader[XGBoostClassificationModel] = new XGBoostClassificationModelReader

  override def load(path: String): XGBoostClassificationModel = super.load(path)

  private[XGBoostClassificationModel]
  class XGBoostClassificationModelWriter(instance: XGBoostClassificationModel)
    extends XGBoostWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)

      // Save model data
      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostClassificationModel")
      val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
      instance._booster.saveModel(outputStream, getModelFormat())
      outputStream.close()
    }
  }

  private class XGBoostClassificationModelReader extends MLReader[XGBoostClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[XGBoostClassificationModel].getName

    override def load(path: String): XGBoostClassificationModel = {
      implicit val sc = super.sparkSession.sparkContext

      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostClassificationModel")
      val dataInStream = internalPath.getFileSystem(sc.hadoopConfiguration).open(internalPath)
      val numClasses = DefaultXGBoostParamsReader.getNumClass(metadata, dataInStream)
      val booster = SXGBoost.loadModel(dataInStream)
      val model = new XGBoostClassificationModel(metadata.uid, numClasses, booster)
      DefaultXGBoostParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
