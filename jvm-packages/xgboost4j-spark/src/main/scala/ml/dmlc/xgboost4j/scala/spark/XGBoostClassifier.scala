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

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.spark.params._
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait, ObjectiveTrait, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats

import scala.collection.JavaConverters._
import scala.collection.{AbstractIterator, Iterator, mutable}

class XGBoostClassifier (
    override val uid: String,
    private val xgboostParams: Map[String, Any])
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

  def setTimeoutRequestWorkers(value: Long): this.type = set(timeoutRequestWorkers, value)

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

  def setSketchEps(value: Double): this.type = set(sketchEps, value)

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

  // called at the start of fit/train when 'eval_metric' is not defined
  private def setupDefaultEvalMetric(): String = {
    require(isDefined(objective), "Users must set \'objective\' via xgboostParams.")
    if ($(objective).startsWith("multi")) {
      // multi
      "merror"
    } else {
      // binary
      "error"
    }
  }

  override protected def train(dataset: Dataset[_]): XGBoostClassificationModel = {

    if (!isDefined(evalMetric) || $(evalMetric).isEmpty) {
      set(evalMetric, setupDefaultEvalMetric())
    }

    if (isDefined(customObj) && $(customObj) != null) {
      set(objectiveType, "classification")
    }

    val _numClasses = getNumClasses(dataset)
    if (isDefined(numClass) && $(numClass) != _numClasses) {
      throw new Exception("The number of classes in dataset doesn't match " +
        "\'num_class\' in xgboost params.")
    }

    val weight = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val baseMargin = if (!isDefined(baseMarginCol) || $(baseMarginCol).isEmpty) {
      lit(Float.NaN)
    } else {
      col($(baseMarginCol))
    }

    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      col($(labelCol)), col($(featuresCol)), weight, baseMargin,
      None, $(numWorkers), needDeterministicRepartitioning, dataset.asInstanceOf[DataFrame]).head
    val evalRDDMap = getEvalSets(xgboostParams).map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(col($(labelCol)), col($(featuresCol)),
          weight, baseMargin, None, $(numWorkers), needDeterministicRepartitioning, dataFrame).head)
    }
    transformSchema(dataset.schema, logging = true)
    val derivedXGBParamMap = MLlib2XGBoostParams
    // All non-null param maps in XGBoostClassifier are in derivedXGBParamMap.
    val (_booster, _metrics) = XGBoost.trainDistributed(trainingSet, derivedXGBParamMap,
      hasGroup = false, evalRDDMap)
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
    private[spark] val _booster: Booster)
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

  def setInferBatchSize(value: Int): this.type = set(inferBatchSize, value)

  /**
   * Single instance prediction.
   * Note: The performance is not ideal, use it carefully!
   */
  override def predict(features: Vector): Double = {
    import DataUtils._
    val dm = new DMatrix(XGBoost.processMissingValues(Iterator(features.asXGB), $(missing)))
    val probability = _booster.predict(data = dm)(0).map(_.toDouble)
    if (numClasses == 2) {
      math.round(probability(0))
    } else {
      probability2prediction(Vectors.dense(probability))
    }
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override protected def predictRaw(features: Vector): Vector = {
    throw new Exception("XGBoost-Spark does not support \'predictRaw\'")
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    throw new Exception("XGBoost-Spark does not support \'raw2probabilityInPlace\'")
  }

  // Generate raw prediction and probability prediction.
  private def transformInternal(dataset: Dataset[_]): DataFrame = {

    val schema = StructType(dataset.schema.fields ++
      Seq(StructField(name = _rawPredictionCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)) ++
      Seq(StructField(name = _probabilityCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)))

    val bBooster = dataset.sparkSession.sparkContext.broadcast(_booster)
    val appName = dataset.sparkSession.sparkContext.appName

    val resultRDD = dataset.asInstanceOf[Dataset[Row]].rdd.mapPartitions { rowIterator =>
      new AbstractIterator[Row] {
        private var batchCnt = 0

        private val batchIterImpl = rowIterator.grouped($(inferBatchSize)).flatMap { batchRow =>
          if (batchCnt == 0) {
            val rabitEnv = Array(
              "DMLC_TASK_ID" -> TaskContext.getPartitionId().toString,
              "DMLC_WORKER_STOP_PROCESS_ON_ERROR" -> "false").toMap
            Rabit.init(rabitEnv.asJava)
          }

          val features = batchRow.iterator.map(row => row.getAs[Vector]($(featuresCol)))

          import DataUtils._
          val cacheInfo = {
            if ($(useExternalMemory)) {
              s"$appName-${TaskContext.get().stageId()}-dtest_cache-" +
                  s"${TaskContext.getPartitionId()}-batch-$batchCnt"
            } else {
              null
            }
          }

          val dm = new DMatrix(
            XGBoost.processMissingValues(features.map(_.asXGB), $(missing)),
            cacheInfo)
          try {
            val Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr) =
              producePredictionItrs(bBooster, dm)
            produceResultIterator(batchRow.iterator,
              rawPredictionItr, probabilityItr, predLeafItr, predContribItr)
          } finally {
            batchCnt += 1
            dm.delete()
          }
        }

        override def hasNext: Boolean = batchIterImpl.hasNext

        override def next(): Row = {
          val ret = batchIterImpl.next()
          if (!batchIterImpl.hasNext) {
            Rabit.shutdown()
          }
          ret
        }
      }
    }

    bBooster.unpersist(blocking = false)
    dataset.sparkSession.createDataFrame(resultRDD, generateResultSchema(schema))
  }

  private def produceResultIterator(
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

  private def generateResultSchema(fixedSchema: StructType): StructType = {
    var resultSchema = fixedSchema
    if (isDefined(leafPredictionCol)) {
      resultSchema = resultSchema.add(StructField(name = $(leafPredictionCol), dataType =
        ArrayType(FloatType, containsNull = false), nullable = false))
    }
    if (isDefined(contribPredictionCol)) {
      resultSchema = resultSchema.add(StructField(name = $(contribPredictionCol), dataType =
        ArrayType(FloatType, containsNull = false), nullable = false))
    }
    resultSchema
  }

  private def producePredictionItrs(broadcastBooster: Broadcast[Booster], dm: DMatrix):
      Array[Iterator[Row]] = {
    val rawPredictionItr = {
      broadcastBooster.value.predict(dm, outPutMargin = true, $(treeLimit)).
        map(Row(_)).iterator
    }
    val probabilityItr = {
      broadcastBooster.value.predict(dm, outPutMargin = false, $(treeLimit)).
        map(Row(_)).iterator
    }
    val predLeafItr = {
      if (isDefined(leafPredictionCol)) {
        broadcastBooster.value.predictLeaf(dm, $(treeLimit)).map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    val predContribItr = {
      if (isDefined(contribPredictionCol)) {
        broadcastBooster.value.predictContrib(dm, $(treeLimit)).map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr)
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
    var outputData = transformInternal(dataset)
    var numColsOutput = 0

    val rawPredictionUDF = udf { rawPrediction: mutable.WrappedArray[Float] =>
      val raw = rawPrediction.map(_.toDouble).toArray
      val rawPredictions = if (numClasses == 2) Array(-raw(0), raw(0)) else raw
      Vectors.dense(rawPredictions)
    }

    val probabilityUDF = udf { probability: mutable.WrappedArray[Float] =>
      val prob = probability.map(_.toDouble).toArray
      val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
      Vectors.dense(probabilities)
    }

    val predictUDF = udf { probability: mutable.WrappedArray[Float] =>
      // From XGBoost probability to MLlib prediction
      val prob = probability.map(_.toDouble).toArray
      val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
      probability2prediction(Vectors.dense(probabilities))
    }

    if ($(rawPredictionCol).nonEmpty) {
      outputData = outputData
        .withColumn(getRawPredictionCol, rawPredictionUDF(col(_rawPredictionCol)))
      numColsOutput += 1
    }

    if ($(probabilityCol).nonEmpty) {
      outputData = outputData
        .withColumn(getProbabilityCol, probabilityUDF(col(_probabilityCol)))
      numColsOutput += 1
    }

    if ($(predictionCol).nonEmpty) {
      outputData = outputData
        .withColumn($(predictionCol), predictUDF(col(_probabilityCol)))
      numColsOutput += 1
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

  private val _rawPredictionCol = "_rawPrediction"
  private val _probabilityCol = "_probability"

  override def read: MLReader[XGBoostClassificationModel] = new XGBoostClassificationModelReader

  override def load(path: String): XGBoostClassificationModel = super.load(path)

  private[XGBoostClassificationModel]
  class XGBoostClassificationModelWriter(instance: XGBoostClassificationModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      implicit val format = DefaultFormats
      implicit val sc = super.sparkSession.sparkContext

      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
      // Save model data
      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostClassificationModel")
      val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
      outputStream.writeInt(instance.numClasses)
      instance._booster.saveModel(outputStream)
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
      val numClasses = dataInStream.readInt()

      val booster = SXGBoost.loadModel(dataInStream)
      val model = new XGBoostClassificationModel(metadata.uid, numClasses, booster)
      DefaultXGBoostParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
