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

import scala.collection.JavaConverters._
import scala.collection.mutable

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.spark.params._
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.json4s.DefaultFormats

private[spark] trait XGBoostClassifierParams extends LearningTaskParams
  with GeneralParams with BoosterParams with HasWeightCol with HasBaseMarginCol

class XGBoostClassifier (
    override val uid: String,
    private val xgboostParams: Map[String, Any])
  extends ProbabilisticClassifier[Vector, XGBoostClassifier, XGBoostClassificationModel]
    with XGBoostClassifierParams with MLWritable {

  def this() = this(Identifiable.randomUID("xgbc"), Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(
    Identifiable.randomUID("xgbc"), xgboostParams)

  /**
   * number of classes
   */
  final val numClasses = new IntParam(this, "num_class", "number of classes")
  setDefault(numClasses, 2)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  private def fromXGBParamMapToParams(): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      params.find(_.name == paramName) match {
        case None =>
        case Some(_: DoubleParam) =>
          set(paramName, paramValue.toString.toDouble)
        case Some(_: BooleanParam) =>
          set(paramName, paramValue.toString.toBoolean)
        case Some(_: IntParam) =>
          set(paramName, paramValue.toString.toInt)
        case Some(_: FloatParam) =>
          set(paramName, paramValue.toString.toFloat)
        case Some(_: Param[_]) =>
          set(paramName, paramValue)
      }
    }
  }

  fromXGBParamMapToParams()

  private[spark] def fromParamsToXGBParamMap: Map[String, Any] = {
    val xgbParamMap = new mutable.HashMap[String, Any]()
    for (param <- params) {
      if (isDefined(param)) {
        xgbParamMap += param.name -> $(param)
      }
    }
    val r = xgbParamMap.toMap
    if ($(numClasses) == 2) {
      r - "num_class"
    } else {
      r
    }
  }

  // called at the start of fit/train when 'eval_metric' is not defined
  private def setupDefaultEvalMetric(): String = {
    val objFunc = xgboostParams("objective")
    require(objFunc != null, "Users must set \'objective\' via xgboostParams.")
    if (objFunc.toString.startsWith("multi")) {
      // multi
      "merror"
    } else {
      // binary
      "error"
    }
  }

  override protected def train(dataset: Dataset[_]): XGBoostClassificationModel = {

    if (xgboostParams.get("eval_metric").isEmpty) {
      set("eval_metric", setupDefaultEvalMetric())
    }

    val weight = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val baseMargin = if (!isDefined(baseMarginCol) || $(baseMarginCol).isEmpty) {
      lit(Float.NaN)
    } else {
      col($(baseMarginCol))
    }

    val instances: RDD[XGBLabeledPoint] = dataset.select(
      col($(featuresCol)),
      col($(labelCol)).cast(FloatType),
      baseMargin.cast(FloatType),
      weight.cast(FloatType)
    ).rdd.map { case Row(features: Vector, label: Float, baseMargin: Float, weight: Float) =>
      val (indices, values) = features match {
        case v: SparseVector => (v.indices, v.values.map(_.toFloat))
        case v: DenseVector => (null, v.values.map(_.toFloat))
      }
      XGBLabeledPoint(label, indices, values, baseMargin = baseMargin, weight = weight)
    }
    transformSchema(dataset.schema, logging = true)
    val derivedXGBParamMap = fromParamsToXGBParamMap
    // All non-null param maps in XGBoostClassifier are in derivedXGBParamMap.
    val (booster, metrics) = XGBoost.trainDistributed(instances, derivedXGBParamMap,
      $(round), $(nWorkers), $(customObj), $(customEval), $(useExternalMemory),
      $(missing))
    val model = new XGBoostClassificationModel(uid, $(numClasses), booster)
    val summary = XGBoostTrainingSummary(metrics)
    model.setSummary(summary)
    model
  }

  override def copy(extra: ParamMap): XGBoostClassifier = defaultCopy(extra)

  override def write: MLWriter = new XGBoostClassifier.XGBoostClassifierWriter(this)
}

object XGBoostClassifier extends MLReadable[XGBoostClassifier] {

  override def read: MLReader[XGBoostClassifier] = new XGBoostClassifierReader

  override def load(path: String): XGBoostClassifier = super.load(path)

  private[XGBoostClassifier]
  class XGBoostClassifierWriter(instance: XGBoostClassifier) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      require(instance.fromParamsToXGBParamMap("custom_eval") == null &&
        instance.fromParamsToXGBParamMap("custom_obj") == null,
        "we do not support persist XGBoostEstimator with customized evaluator and objective" +
          " function for now")
      implicit val format = DefaultFormats
      implicit val sc = super.sparkSession.sparkContext
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
    }
  }

  private class XGBoostClassifierReader extends MLReader[XGBoostClassifier] {

    override def load(path: String): XGBoostClassifier = {
      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
      val cls = Utils.classForName(metadata.className)
      val instance =
        cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
      DefaultXGBoostParamsReader.getAndSetParams(instance, metadata)
      instance.asInstanceOf[XGBoostClassifier]
    }
  }
}

class XGBoostClassificationModel private[ml](
    override val uid: String,
    override val numClasses: Int,
    private[spark] val _booster: Booster)
  extends ProbabilisticClassificationModel[Vector, XGBoostClassificationModel]
    with XGBoostClassifierParams with MLWritable with Serializable {

  import XGBoostClassificationModel._

  // only called in copy()
  def this(uid: String) = this(uid, 2, null)

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

  // TODO: Make it public after we resolve performance issue
  private def margin(features: Vector): Array[Float] = {
    import DataUtils._
    val dm = new DMatrix(scala.collection.Iterator(features.asXGB))
    _booster.predict(data = dm, outPutMargin = true)(0)
  }

  private def probability(features: Vector): Array[Float] = {
    import DataUtils._
    val dm = new DMatrix(scala.collection.Iterator(features.asXGB))
    _booster.predict(data = dm, outPutMargin = false)(0)
  }

  // The performance of single instance prediction is poor, need to be optimized later.
  override def predict(features: Vector): Double = {
    val _probability = probability(features)
    val probabilityVec = if (numClasses == 2) {
      Vectors.dense(Array(1.0 - _probability(0), _probability(0)))
    } else {
      Vectors.dense(_probability.map(_.toDouble))
    }
    probabilityVec.argmax
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override def predictRaw(features: Vector): Vector = {
    Vectors.dense(margin(features).map(_.toDouble))
  }

  // Actually we don't use this function at all, to make it pass compiler check.
  override def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction
  }

  // Generate raw prediction and probability prediction.
  def transformInternal(dataset: Dataset[_]): DataFrame = {

    val schema = StructType(dataset.schema.fields ++
      Seq(StructField(name = _rawPredictionCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)) ++
      Seq(StructField(name = _probabilityCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)))

    val bBooster = dataset.sparkSession.sparkContext.broadcast(_booster)
    val appName = dataset.sparkSession.sparkContext.appName

    val rdd = dataset.rdd.mapPartitions { rowIterator =>
      if (rowIterator.hasNext) {
        val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
        Rabit.init(rabitEnv.asJava)
        val (rowItr1, rowItr2) = rowIterator.duplicate
        val featuresIterator = rowItr2.map(row => row.asInstanceOf[Row].getAs[Vector](
          $(featuresCol))).toList.iterator
        import DataUtils._
        val cacheInfo = {
          if ($(useExternalMemory)) {
            s"$appName-${TaskContext.get().stageId()}-dtest_cache-${TaskContext.getPartitionId()}"
          } else {
            null
          }
        }

        val dm = new DMatrix(featuresIterator.map(_.asXGB), cacheInfo)
        try {
          val rawPredictionItr = {
            bBooster.value.predict(dm, outPutMargin = true).map(Row(_)).iterator
          }
          val probabilityItr = {
            bBooster.value.predict(dm, outPutMargin = false).map(Row(_)).iterator
          }
          Rabit.shutdown()
          rowItr1.zip(rawPredictionItr).zip(probabilityItr).map {
            case ((originals: Row, rawPrediction: Row), probability: Row) =>
              Row.fromSeq(originals.toSeq ++ rawPrediction.toSeq ++ probability.toSeq)
          }
        } finally {
          dm.delete()
        }
      } else {
        Iterator[Row]()
      }
    }

    bBooster.unpersist(blocking = false)

    dataset.sparkSession.createDataFrame(rdd, schema)
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

    val rawPredictionUDF = udf { (rawPrediction: mutable.WrappedArray[Float]) =>
      Vectors.dense(rawPrediction.map(_.toDouble).toArray)
    }

    val probabilityUDF = udf { (probability: mutable.WrappedArray[Float]) =>
      if (numClasses == 2) {
        Vectors.dense(Array(1 - probability(0), probability(0)).map(_.toDouble))
      } else {
        Vectors.dense(probability.map(_.toDouble).toArray)
      }
    }

    val predictUDF = udf { (probability: mutable.WrappedArray[Float]) =>
      // From XGBoost probability to MLlib prediction
      val probabilities = if (numClasses == 2) {
        Array(1 - probability(0), probability(0)).map(_.toDouble)
      } else {
        probability.map(_.toDouble).toArray
      }
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
