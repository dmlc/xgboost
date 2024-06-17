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

package ml.dmlc.xgboost4j.scala.spark

import java.util.ServiceLoader

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.iterableAsScalaIterableConverter

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.spark.Utils.MLVectorToXGBLabeledPoint
import ml.dmlc.xgboost4j.scala.spark.params._


/**
 * Hold the column indexes used to get the column index
 */
private case class ColumnIndexes(label: String,
                                 features: String,
                                 weight: Option[String] = None,
                                 baseMargin: Option[String] = None,
                                 group: Option[String] = None,
                                 valiation: Option[String] = None)

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M],
  M <: XGBoostModel[M]
] extends Estimator[M] with XGBoostParams[Learner] with SparkParams[Learner]
  with ParamMapConversion with DefaultParamsWritable {

  protected val logger = LogFactory.getLog("XGBoostSpark")

  // Find the XGBoostPlugin by ServiceLoader
  private val plugin: Option[XGBoostPlugin] = {
    val classLoader = Option(Thread.currentThread().getContextClassLoader)
      .getOrElse(getClass.getClassLoader)

    val serviceLoader = ServiceLoader.load(classOf[XGBoostPlugin], classLoader)

    // For now, we only trust GPUXGBoostPlugin.
    serviceLoader.asScala.filter(x => x.getClass.getName.equals(
      "ml.dmlc.xgboost4j.scala.spark.GPUXGBoostPlugin")).toList match {
      case Nil => None
      case head :: Nil =>
        Some(head)
      case _ => None
    }
  }

  private def isPluginEnabled(dataset: Dataset[_]): Boolean = {
    plugin.map(_.isEnabled(dataset)).getOrElse(false)
  }

  /**
   * Pre-convert input double data to floats to align with XGBoost's internal float-based
   * operations to save memory usage.
   *
   * @param dataset the input dataset
   * @param name    which column will be casted to float if possible.
   * @return Dataset
   */
  private def castToFloatIfNeeded(schema: StructType, name: String): Column = {
    if (!schema(name).dataType.isInstanceOf[FloatType]) {
      val meta = schema(name).metadata
      col(name).as(name, meta).cast(FloatType)
    } else {
      col(name)
    }
  }

  /**
   * Preprocess the dataset to meet the xgboost input requirement
   *
   * @param dataset
   * @return
   */
  private def preprocess(dataset: Dataset[_]): (Dataset[_], ColumnIndexes) = {
    // Columns to be selected for XGBoost
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    // TODO, support columnar and array.
    selectedCols.append(castToFloatIfNeeded(schema, getLabelCol))
    selectedCols.append(col(getFeaturesCol))

    val weightName = if (isDefined(weightCol) && getWeightCol.nonEmpty) {
      selectedCols.append(castToFloatIfNeeded(schema, getWeightCol))
      Some(getWeightCol)
    } else {
      None
    }

    val baseMarginName = if (isDefined(baseMarginCol) && getBaseMarginCol.nonEmpty) {
      selectedCols.append(castToFloatIfNeeded(schema, getBaseMarginCol))
      Some(getBaseMarginCol)
    } else {
      None
    }

    // TODO, check the validation col
    val validationName = if (isDefined(validationIndicatorCol) &&
      getValidationIndicatorCol.nonEmpty) {
      selectedCols.append(col(getValidationIndicatorCol))
      Some(getValidationIndicatorCol)
    } else {
      None
    }

    var groupName: Option[String] = None
    this match {
      case p: HasGroupCol =>
        // Cast group col to IntegerType if necessary
        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          selectedCols.append(castToFloatIfNeeded(schema, p.getGroupCol))
          groupName = Some(p.getGroupCol)
        }
      case _ =>
    }

    var input = dataset.select(selectedCols: _*)

    // TODO,
    //  1. add a parameter to force repartition,
    //  2. follow xgboost pyspark way check if repartition is needed.
    val numWorkers = getNumWorkers
    val numPartitions = dataset.rdd.getNumPartitions
    input = if (numWorkers == numPartitions) {
      input
    } else {
      input.repartition(numWorkers)
    }
    val columnIndexes = ColumnIndexes(
      getLabelCol,
      getFeaturesCol,
      weight = weightName,
      baseMargin = baseMarginName,
      group = groupName,
      valiation = validationName)
    (input, columnIndexes)
  }

  /**
   * Convert the dataframe to RDD
   *
   * @param dataset
   * @param columnsOrder the order of columns including weight/group/base margin ...
   * @return RDD
   */
  def toRdd(dataset: Dataset[_], columnIndexes: ColumnIndexes): RDD[Watches] = {

    // 1. to XGBLabeledPoint
    val labeledPointRDD = dataset.rdd.map {
      case row: Row =>
        val label = row.getFloat(row.fieldIndex(columnIndexes.label))
        val features = row.getAs[Vector](columnIndexes.features)
        val weight = columnIndexes.weight.map(v => row.getFloat(row.fieldIndex(v))).getOrElse(1.0f)
        val baseMargin = columnIndexes.baseMargin.map(v =>
          row.getFloat(row.fieldIndex(v))).getOrElse(Float.NaN)
        val group = columnIndexes.group.map(v =>
          row.getFloat(row.fieldIndex(v))).getOrElse(-1.0f)

        // TODO support sparse vector.
        // TODO support array
        val values = features.toArray.map(_.toFloat)
        val isValidation = columnIndexes.valiation.exists(v =>
          row.getBoolean(row.fieldIndex(v)))

        (isValidation,
          XGBLabeledPoint(label, values.length, null, values, weight, group.toInt, baseMargin))
    }


    labeledPointRDD.mapPartitions { iter =>
      val datasets: ArrayBuffer[DMatrix] = ArrayBuffer.empty
      val names: ArrayBuffer[String] = ArrayBuffer.empty
      val validations: ArrayBuffer[XGBLabeledPoint] = ArrayBuffer.empty

      val trainIter = if (columnIndexes.valiation.isDefined) {
        // Extract validations during build Train DMatrix
        val dataIter = new Iterator[XGBLabeledPoint] {
          private var tmp: Option[XGBLabeledPoint] = None

          override def hasNext: Boolean = {
            if (tmp.isDefined) {
              return true
            }
            while (iter.hasNext) {
              val (isVal, labelPoint) = iter.next()
              if (isVal) {
                validations.append(labelPoint)
              } else {
                tmp = Some(labelPoint)
                return true
              }
            }
            false
          }

          override def next(): XGBLabeledPoint = {
            val xgbLabeledPoint = tmp.get
            tmp = None
            xgbLabeledPoint
          }
        }
        dataIter
      } else {
        iter.map(_._2)
      }

      datasets.append(new DMatrix(trainIter))
      names.append(Utils.TRAIN_NAME)
      if (columnIndexes.valiation.isDefined) {
        datasets.append(new DMatrix(validations.toIterator))
        names.append(Utils.VALIDATION_NAME)
      }

      // TODO  1. support external memory 2. rework or remove Watches
      val watches = new Watches(datasets.toArray, names.toArray, None)
      Iterator.single(watches)
    }
  }

  protected def createModel(booster: Booster, summary: XGBoostTrainingSummary): M

  private def getRuntimeParameters(isLocal: Boolean): RuntimeParams = {

    val runOnGpu = false

    RuntimeParams(
      getNumWorkers,
      getNumRound,
      null, // TODO support ObjectiveTrait
      null, // TODO support EvalTrait
      TrackerConf(getRabitTrackerTimeout, getRabitTrackerHostIp, getRabitTrackerPort),
      getNumEarlyStoppingRounds,
      getDevice,
      isLocal,
      runOnGpu
    )
  }

  /**
   * Check to see if Spark expects SSL encryption (`spark.ssl.enabled` set to true).
   * If so, throw an exception unless this safety measure has been explicitly overridden
   * via conf `xgboost.spark.ignoreSsl`.
   */
  private def validateSparkSslConf(spark: SparkSession): Unit = {

    val sparkSslEnabled = spark.conf.getOption("spark.ssl.enabled").getOrElse("false").toBoolean
    val xgbIgnoreSsl = spark.conf.getOption("xgboost.spark.ignoreSsl").getOrElse("false").toBoolean

    if (sparkSslEnabled) {
      if (xgbIgnoreSsl) {
        logger.warn(s"spark-xgboost is being run without encrypting data in transit!  " +
          s"Spark Conf spark.ssl.enabled=true was overridden with xgboost.spark.ignoreSsl=true.")
      } else {
        throw new Exception("xgboost-spark found spark.ssl.enabled=true to encrypt data " +
          "in transit, but xgboost-spark sends non-encrypted data over the wire for efficiency. " +
          "To override this protection and still use xgboost-spark at your own risk, " +
          "you can set the SparkSession conf to use xgboost.spark.ignoreSsl=true.")
      }
    }
  }

  /**
   * Validate the parameters before training, throw exception if possible
   */
  protected def validate(dataset: Dataset[_]): Unit = {
    validateSparkSslConf(dataset.sparkSession)
    val schema = dataset.schema
    SparkUtils.checkNumericType(schema, $(labelCol))
    if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      SparkUtils.checkNumericType(schema, $(weightCol))
    }

    if (isDefined(baseMarginCol) && $(baseMarginCol).nonEmpty) {
      SparkUtils.checkNumericType(schema, $(baseMarginCol))
    }

    // TODO Move this to XGBoostRanker
    //    this match {
    //      case p: HasGroupCol =>
    //        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
    //          SparkUtils.checkNumericType(schema, p.getGroupCol)
    //        }
    //    }

    val taskCpus = dataset.sparkSession.sparkContext.getConf.getInt("spark.task.cpus", 1)
    if (isDefined(nthread)) {
      if (getNthread > taskCpus) {
        logger.warn("nthread must be smaller or equal to spark.task.cpus.")
        setNthread(taskCpus)
      }
    } else {
      setNthread(taskCpus)
    }

  }

  override def fit(dataset: Dataset[_]): M = {
    validate(dataset)

    val rdd = if (isPluginEnabled(dataset)) {
      plugin.get.buildRddWatches(this, dataset)
    } else {
      val (input, columnIndexes) = preprocess(dataset)
      toRdd(input, columnIndexes)
    }

    val xgbParams = getXGBoostParams

    val runtimeParams = getRuntimeParameters(dataset.sparkSession.sparkContext.isLocal)

    val (booster, metrics) = XGBoost.train(rdd, runtimeParams, xgbParams)

    val summary = XGBoostTrainingSummary(metrics)
    copyValues(createModel(booster, summary))
  }

  override def copy(extra: ParamMap): Learner = defaultCopy(extra)

  // Not used in XGBoost
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, true)
  }
}

/**
 * XGBoost base model
 *
 * @param uid
 * @param model           xgboost booster
 * @param trainingSummary the training summary
 * @tparam the exact model which must extend from XGBoostModel
 */
private[spark] abstract class XGBoostModel[M <: XGBoostModel[M]](
  override val uid: String,
  private val model: Booster,
  private val trainingSummary: Option[XGBoostTrainingSummary]) extends Model[M] with MLWritable
  with XGBoostParams[M] with SparkParams[M] {

  protected val TMP_TRANSFORMED_COL = "_tmp_xgb_transformed_col"

  override def copy(extra: ParamMap): M = defaultCopy(extra).asInstanceOf[M]

  /**
   * Get the native XGBoost Booster
   *
   * @return
   */
  def nativeBooster: Booster = model

  def summary: XGBoostTrainingSummary = trainingSummary.getOrElse {
    throw new IllegalStateException("No training summary available for this XGBoostModel")
  }

  // Not used in XGBoost
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, false)
  }

  def postTransform(dataset: Dataset[_]): Dataset[_] = dataset

  override def transform(dataset: Dataset[_]): DataFrame = {

    val spark = dataset.sparkSession

    // Be careful about the order of columns
    var schema = dataset.schema

    var hasLeafPredictionCol = false
    if (isDefined(leafPredictionCol) && getLeafPredictionCol.nonEmpty) {
      schema = schema.add(StructField(getLeafPredictionCol, ArrayType(FloatType)))
      hasLeafPredictionCol = true
    }

    var hasContribPredictionCol = false
    if (isDefined(contribPredictionCol) && getContribPredictionCol.nonEmpty) {
      schema = schema.add(StructField(getContribPredictionCol, ArrayType(FloatType)))
      hasContribPredictionCol = true
    }

    var hasRawPredictionCol = false
    // For classification case, the tranformed col is probability,
    // while for others, it's the prediction value.
    var hasTransformedCol = false
    this match {
      case p: ClassificationParams[_] => // classification case
        if (isDefined(p.rawPredictionCol) && p.getRawPredictionCol.nonEmpty) {
          schema = schema.add(
            StructField(p.getRawPredictionCol, ArrayType(FloatType)))
          hasRawPredictionCol = true
        }
        if (isDefined(p.probabilityCol) && p.getProbabilityCol.nonEmpty) {
          schema = schema.add(
            StructField(TMP_TRANSFORMED_COL, ArrayType(FloatType)))
          hasTransformedCol = true
        }

        if (isDefined(predictionCol) && getPredictionCol.nonEmpty) {
          // Let's use transformed col to calculate the prediction
          if (!hasTransformedCol) {
            // Add the transformed col for predition
            schema = schema.add(
              StructField(TMP_TRANSFORMED_COL, ArrayType(FloatType)))
            hasTransformedCol = true
          }
        }
      case _ =>
        // Rename TMP_TRANSFORMED_COL to prediction in the postTransform.
        if (isDefined(predictionCol) && getPredictionCol.nonEmpty) {
          schema = schema.add(
            StructField(TMP_TRANSFORMED_COL, ArrayType(FloatType)))
          hasTransformedCol = true
        }
    }

    // TODO configurable
    val inferBatchSize = 32 << 10
    // Broadcast the booster to each executor.
    val bBooster = spark.sparkContext.broadcast(nativeBooster)
    val featureName = getFeaturesCol

    val outputData = dataset.toDF().mapPartitions { rowIter =>

      rowIter.grouped(inferBatchSize).flatMap { batchRow =>
        val features = batchRow.iterator.map(row => row.getAs[Vector](
          row.fieldIndex(featureName)))

        // DMatrix used to prediction
        val dm = new DMatrix(features.map(_.asXGB))

        var tmpOut = batchRow.map(_.toSeq)

        val zip = (left: Seq[Seq[_]], right: Array[Array[Float]]) => left.zip(right).map {
          case (a, b) => a ++ Seq(b)
        }

        if (hasLeafPredictionCol) {
          tmpOut = zip(tmpOut, bBooster.value.predictLeaf(dm))
        }
        if (hasContribPredictionCol) {
          tmpOut = zip(tmpOut, bBooster.value.predictContrib(dm))
        }
        if (hasRawPredictionCol) {
          tmpOut = zip(tmpOut, bBooster.value.predict(dm, outPutMargin = true))
        }
        if (hasTransformedCol) {
          tmpOut = zip(tmpOut, bBooster.value.predict(dm, outPutMargin = false))
        }
        tmpOut.map(Row.fromSeq)
      }

    }(Encoders.row(schema))
    bBooster.unpersist(blocking = false)
    postTransform(outputData).toDF()
  }

  override def write: MLWriter = new XGBoostModelWriter[XGBoostModel[_]](this)
}

/**
 * Class to write the model
 *
 * @param instance model to be written
 */
private[spark] class XGBoostModelWriter[M <: XGBoostModel[M]](instance: M) extends MLWriter {
  override protected def saveImpl(path: String): Unit = {
    SparkUtils.saveMetadata(instance, path, sc)

    // Save model data
    val dataPath = new Path(path, "data").toString
    val internalPath = new Path(dataPath, "model")
    val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
    try {
      instance.nativeBooster.saveModel(outputStream)
    } finally {
      outputStream.close()
    }
  }
}

private[spark] abstract class XGBoostModelReader[M <: XGBoostModel[M]] extends MLReader[M] {

  protected def loadBooster(path: String): Booster = {
    val dataPath = new Path(path, "data").toString
    val internalPath = new Path(dataPath, "model")
    val dataInStream = internalPath.getFileSystem(sc.hadoopConfiguration).open(internalPath)
    try {
      SXGBoost.loadModel(dataInStream)
    } finally {
      dataInStream.close()
    }
  }
}
