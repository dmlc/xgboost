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

import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.spark.Utils.MLVectorToXGBLabeledPoint
import ml.dmlc.xgboost4j.scala.spark.params._


/**
 * Hold the column index
 */
private[spark] case class ColumnIndices(
    labelId: Int,
    featureId: Option[Int], // the feature type is VectorUDT or Array
    featureIds: Option[Seq[Int]], // the feature type is columnar
    weightId: Option[Int],
    marginId: Option[Int],
    groupId: Option[Int])

private[spark] trait NonParamVariables[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]] {

  private var dataset: Option[Dataset[_]] = None

  def setEvalDataset(ds: Dataset[_]): T = {
    this.dataset = Some(ds)
    this.asInstanceOf[T]
  }

  def getEvalDataset(): Option[Dataset[_]] = {
    this.dataset
  }
}

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M], M <: XGBoostModel[M]] extends Estimator[M]
  with XGBoostParams[Learner] with SparkParams[Learner]
  with NonParamVariables[Learner, M] with ParamMapConversion with DefaultParamsWritable {

  protected val logger = LogFactory.getLog("XGBoostSpark")

  // Find the XGBoostPlugin by ServiceLoader
  private val plugin: Option[XGBoostPlugin] = {
    val classLoader = Option(Thread.currentThread().getContextClassLoader)
      .getOrElse(getClass.getClassLoader)

    val serviceLoader = ServiceLoader.load(classOf[XGBoostPlugin], classLoader)

    // For now, we only trust GpuXGBoostPlugin.
    serviceLoader.asScala.filter(x => x.getClass.getName.equals(
      "ml.dmlc.xgboost4j.scala.spark.GpuXGBoostPlugin")).toList match {
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
  private[spark] def castToFloatIfNeeded(schema: StructType, name: String): Column = {
    if (!schema(name).dataType.isInstanceOf[FloatType]) {
      val meta = schema(name).metadata
      col(name).as(name, meta).cast(FloatType)
    } else {
      col(name)
    }
  }

  /**
   * Repartition the dataset to the numWorkers if needed.
   *
   * @param dataset to be repartition
   * @return the repartitioned dataset
   */
  private[spark] def repartitionIfNeeded(dataset: Dataset[_]): Dataset[_] = {
    val numPartitions = dataset.rdd.getNumPartitions
    if (getForceRepartition || getNumWorkers != numPartitions) {
      dataset.repartition(getNumWorkers)
    } else {
      dataset
    }
  }

  /**
   * Build the columns indices.
   */
  private[spark] def buildColumnIndices(schema: StructType): ColumnIndices = {
    // Get feature id(s)
    val (featureIds: Option[Seq[Int]], featureId: Option[Int]) =
      if (getFeaturesCols.length != 0) {
        (Some(getFeaturesCols.map(schema.fieldIndex).toSeq), None)
      } else {
        (None, Some(schema.fieldIndex(getFeaturesCol)))
      }

    // function to get the column id according to the parameter
    def columnId(param: Param[String]): Option[Int] = {
      if (isDefined(param) && $(param).nonEmpty) {
        Some(schema.fieldIndex($(param)))
      } else {
        None
      }
    }

    // Special handle for group
    val groupId: Option[Int] = this match {
      case p: HasGroupCol => columnId(p.groupCol)
      case _ => None
    }

    ColumnIndices(
      labelId = columnId(labelCol).get,
      featureId = featureId,
      featureIds = featureIds,
      columnId(weightCol),
      columnId(baseMarginCol),
      groupId)
  }

  private[spark] def isDefinedNonEmpty(param: Param[String]): Boolean = {
    if (isDefined(param) && $(param).nonEmpty) true else false
  }

  /**
   * Preprocess the dataset to meet the xgboost input requirement
   *
   * @param dataset
   * @return
   */
  private[spark] def preprocess(dataset: Dataset[_]): (Dataset[_], ColumnIndices) = {

    // Columns to be selected for XGBoost training
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    def selectCol(c: Param[String]) = {
      if (isDefinedNonEmpty(c)) {
        // Validation col should be a boolean column.
        if (c == featuresCol) {
          selectedCols.append(col($(c)))
        } else {
          selectedCols.append(castToFloatIfNeeded(schema, $(c)))
        }
      }
    }

    Seq(labelCol, featuresCol, weightCol, baseMarginCol).foreach(selectCol)
    this match {
      case p: HasGroupCol => selectCol(p.groupCol)
      case _ =>
    }
    val input = repartitionIfNeeded(dataset.select(selectedCols: _*))

    val columnIndices = buildColumnIndices(input.schema)
    (input, columnIndices)
  }

  /** visible for testing */
  private[spark] def toXGBLabeledPoint(dataset: Dataset[_],
                                       columnIndexes: ColumnIndices): RDD[XGBLabeledPoint] = {
    val missing = getMissing
    dataset.toDF().rdd.mapPartitions { input: Iterator[Row] =>

      def isMissing(values: Array[Double]): Boolean = {
        if (missing.isNaN) {
          values.exists(_.toFloat.isNaN)
        } else {
          values.exists(_.toFloat == missing)
        }
      }

      new Iterator[XGBLabeledPoint] {
        private var tmp: Option[XGBLabeledPoint] = None

        override def hasNext: Boolean = {
          if (tmp.isDefined) {
            return true
          }
          while (input.hasNext) {
            val row = input.next()
            val features = row.getAs[Vector](columnIndexes.featureId.get)
            if (!isMissing(features.toArray)) {
              val label = row.getFloat(columnIndexes.labelId)
              val weight = columnIndexes.weightId.map(row.getFloat).getOrElse(1.0f)
              val baseMargin = columnIndexes.marginId.map(row.getFloat).getOrElse(Float.NaN)
              val group = columnIndexes.groupId.map(row.getFloat).getOrElse(-1.0f)
              val (size, indices, values) = features match {
                case v: SparseVector => (v.size, v.indices, v.values.map(_.toFloat))
                case v: DenseVector => (v.size, null, v.values.map(_.toFloat))
              }
              tmp = Some(XGBLabeledPoint(label, size, indices, values, weight,
                group.toInt, baseMargin))
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
    }
  }

  /**
   * Convert the dataframe to RDD, visible to testing
   *
   * @param dataset
   * @param columnsOrder the order of columns including weight/group/base margin ...
   * @return RDD
   */
  private[spark] def toRdd(dataset: Dataset[_], columnIndices: ColumnIndices): RDD[Watches] = {
    val trainRDD = toXGBLabeledPoint(dataset, columnIndices)

    getEvalDataset().map { eval =>
      val (evalDf, _) = preprocess(eval)
      val evalRDD = toXGBLabeledPoint(evalDf, columnIndices)
      trainRDD.zipPartitions(evalRDD) { (trainIter, evalIter) =>
        val trainDMatrix = new DMatrix(trainIter)
        val evalDMatrix = new DMatrix(evalIter)
        val watches = new Watches(Array(trainDMatrix, evalDMatrix),
          Array(Utils.TRAIN_NAME, Utils.VALIDATION_NAME), None)
        Iterator.single(watches)
      }
    }.getOrElse(
      trainRDD.mapPartitions { iter =>
        // Handle weight/base margin
        val watches = new Watches(Array(new DMatrix(iter)), Array(Utils.TRAIN_NAME), None)
        Iterator.single(watches)
      }
    )
  }

  protected def createModel(booster: Booster, summary: XGBoostTrainingSummary): M

  private def getRuntimeParameters(isLocal: Boolean): RuntimeParams = {
    val runOnGpu = if (getDevice != "cpu" || getTreeMethod == "gpu_hist") true else false
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
      require(getNthread <= taskCpus,
        s"the nthread configuration ($getNthread) must be no larger than " +
          s"spark.task.cpus ($taskCpus)")
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

    /** If the parameter is defined, add it to schema and turn true */
    def addToSchema(param: Param[String], colName: Option[String] = None): Boolean = {
      if (isDefined(param) && $(param).nonEmpty) {
        val name = colName.getOrElse($(param))
        schema = schema.add(StructField(name, ArrayType(FloatType)))
        true
      } else {
        false
      }
    }

    val hasLeafPredictionCol = addToSchema(leafPredictionCol)
    val hasContribPredictionCol = addToSchema(contribPredictionCol)

    var hasRawPredictionCol = false
    // For classification case, the tranformed col is probability,
    // while for others, it's the prediction value.
    var hasTransformedCol = false
    this match {
      case p: ClassificationParams[_] => // classification case
        hasRawPredictionCol = addToSchema(p.rawPredictionCol)
        hasTransformedCol = addToSchema(p.probabilityCol, Some(TMP_TRANSFORMED_COL))

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
        hasTransformedCol = addToSchema(predictionCol, Some(TMP_TRANSFORMED_COL))

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

        try {
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
        } finally {
          dm.delete()
        }
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
