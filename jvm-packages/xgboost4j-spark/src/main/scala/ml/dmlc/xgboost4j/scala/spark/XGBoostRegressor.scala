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

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.spark.params.{DefaultXGBoostParamsReader, Utils, _}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}

import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats

import scala.collection.mutable

private[spark] trait XGBoostRegressorParams extends LearningTaskParams
  with GeneralParams with BoosterParams with HasWeightCol

class XGBoostRegressor (
    override val uid: String,
    private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostRegressorParams with HasBaseMarginCol with MLWritable {

  def this() = this(Identifiable.randomUID("xgbr"), Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(
    Identifiable.randomUID("xgbr"), xgboostParams)

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
    r
  }

  // called at the start of fit/train when 'eval_metric' is not defined
  private def setupDefaultEvalMetric(): String = {
    val objFunc = xgboostParams("objective")
    require(objFunc != null, "Users must set \'objective\' via xgboostParams.")
    if (objFunc.toString.startsWith("rank")) {
      "map"
    } else {
      "rmse"
    }
  }

  override protected def train(dataset: Dataset[_]): XGBoostRegressionModel = {

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
    // All non-null param maps in XGBoostRegressor are in derivedXGBParamMap.
    val (booster, metrics) = XGBoost.trainDistributed(instances, derivedXGBParamMap,
      $(round), $(nWorkers), $(customObj), $(customEval), $(useExternalMemory),
      $(missing))
    val model = new XGBoostRegressionModel(uid, booster)
    val summary = XGBoostTrainingSummary(metrics)
    model.setSummary(summary)
    model
  }

  override def copy(extra: ParamMap): XGBoostRegressor = defaultCopy(extra)

  override def write: MLWriter = new XGBoostRegressor.XGBoostRegressorWriter(this)
}

object XGBoostRegressor extends MLReadable[XGBoostRegressor] {

  override def read: MLReader[XGBoostRegressor] = new XGBoostRegressorReader

  override def load(path: String): XGBoostRegressor = super.load(path)

  private[XGBoostRegressor]
  class XGBoostRegressorWriter(instance: XGBoostRegressor) extends MLWriter {

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

  private class XGBoostRegressorReader extends MLReader[XGBoostRegressor] {

    override def load(path: String): XGBoostRegressor = {
      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
      val cls = Utils.classForName(metadata.className)
      val instance =
        cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
      DefaultXGBoostParamsReader.getAndSetParams(instance, metadata)
      instance.asInstanceOf[XGBoostRegressor]
    }
  }
}

class XGBoostRegressionModel private[ml] (
    override val uid: String,
    private[spark] val _booster: Booster)
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with XGBoostRegressorParams with MLWritable with Serializable {

  import XGBoostRegressionModel._

  // only called in copy()
  def this(uid: String) = this(uid, null)

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

  override def predict(features: Vector): Double = {
    throw new Exception("XGBoost-Spark does not support online prediction")
  }

  private def transformInternal(dataset: Dataset[_]): DataFrame = {

    val schema = StructType(dataset.schema.fields ++
      Seq(StructField(name = _originalPredictionCol, dataType =
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
          val originalPredictionItr = {
            bBooster.value.predict(dm).map(Row(_)).iterator
          }
          Rabit.shutdown()
          rowItr1.zip(originalPredictionItr).map {
            case (originals: Row, originalPrediction: Row) =>
              Row.fromSeq(originals.toSeq ++ originalPrediction.toSeq)
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

    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = transformInternal(dataset)
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

  private val _originalPredictionCol = "_originalPrediction"

  override def read: MLReader[XGBoostRegressionModel] = new XGBoostRegressionModelReader

  override def load(path: String): XGBoostRegressionModel = super.load(path)

  private[XGBoostRegressionModel]
  class XGBoostRegressionModelWriter(instance: XGBoostRegressionModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      implicit val format = DefaultFormats
      implicit val sc = super.sparkSession.sparkContext
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
      // Save model data
      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostRegressionModel")
      val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
      instance._booster.saveModel(outputStream)
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
