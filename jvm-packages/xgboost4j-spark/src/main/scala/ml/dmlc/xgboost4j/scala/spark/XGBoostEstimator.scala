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

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.xgboost.{SparkUtils, XGBProbabilisticClassifierParams}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.java.{Booster => JBooster}
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

private[spark] trait PluginMixin {
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

  /** Visible for testing */
  protected[spark] def getPlugin: Option[XGBoostPlugin] = plugin

  protected def isPluginEnabled(dataset: Dataset[_]): Boolean = {
    plugin.map(_.isEnabled(dataset)).getOrElse(false)
  }
}

private[spark] trait XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M], M <: XGBoostModel[M]] extends Estimator[M]
  with XGBoostParams[Learner] with SparkParams[Learner] with ParamUtils[Learner]
  with NonParamVariables[Learner, M] with ParamMapConversion with DefaultParamsWritable
  with PluginMixin {

  protected val logger = LogFactory.getLog("XGBoostSpark")

  /**
   * Cast the field in schema to the desired data type.
   *
   * @param dataset    the input dataset
   * @param name       which column will be casted to float if possible.
   * @param targetType the targetd data type
   * @return Dataset
   */
  private[spark] def castIfNeeded(schema: StructType,
                                  name: String,
                                  targetType: DataType = FloatType): Column = {
    if (!(schema(name).dataType == targetType)) {
      val meta = schema(name).metadata
      col(name).as(name, meta).cast(targetType)
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
      if (isDefinedNonEmpty(param)) {
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

    def selectCol(c: Param[String], targetType: DataType) = {
      if (isDefinedNonEmpty(c)) {
        // Validation col should be a boolean column.
        if (c == featuresCol) {
          selectedCols.append(col($(c)))
        } else {
          selectedCols.append(castIfNeeded(schema, $(c), targetType))
        }
      }
    }

    Seq(labelCol, featuresCol, weightCol, baseMarginCol).foreach(p => selectCol(p, FloatType))
    this match {
      case p: HasGroupCol => selectCol(p.groupCol, IntegerType)
      case _ =>
    }
    val input = repartitionIfNeeded(dataset.select(selectedCols.toArray: _*))

    val columnIndices = buildColumnIndices(input.schema)
    (input, columnIndices)
  }

  /** visible for testing */
  private[spark] def toXGBLabeledPoint(dataset: Dataset[_],
                                       columnIndexes: ColumnIndices): RDD[XGBLabeledPoint] = {
    dataset.toDF().rdd.map { row =>
      val features = row.getAs[Vector](columnIndexes.featureId.get)
      val label = row.getFloat(columnIndexes.labelId)
      val weight = columnIndexes.weightId.map(row.getFloat).getOrElse(1.0f)
      val baseMargin = columnIndexes.marginId.map(row.getFloat).getOrElse(Float.NaN)
      val group = columnIndexes.groupId.map(row.getInt).getOrElse(-1)
      // To make "0" meaningful, we convert sparse vector if possible to dense to create DMatrix.
      val values = features.toArray.map(_.toFloat)
      XGBLabeledPoint(label, values.length, null, values, weight, group, baseMargin)
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

    val featureNames = if (getFeatureNames.isEmpty) None else Some(getFeatureNames)
    val featureTypes = if (getFeatureTypes.isEmpty) None else Some(getFeatureTypes)

    val missing = getMissing

    // Transform the labeledpoint to get margins/groups and build DMatrix
    // TODO support basemargin for multiclassification
    // TODO and optimization, move it into JNI.
    def buildDMatrix(iter: Iterator[XGBLabeledPoint]) = {
      val dmatrix = if (columnIndices.marginId.isDefined || columnIndices.groupId.isDefined) {
        val margins = new mutable.ArrayBuilder.ofFloat
        val groups = new mutable.ArrayBuilder.ofInt
        val groupWeights = new mutable.ArrayBuilder.ofFloat
        var prevGroup = -101010
        var prevWeight = -1.0f
        var groupSize = 0
        val transformedIter = iter.map { labeledPoint =>
          if (columnIndices.marginId.isDefined) {
            margins += labeledPoint.baseMargin
          }
          if (columnIndices.groupId.isDefined) {
            if (prevGroup != labeledPoint.group) {
              // starting with new group
              if (prevGroup != -101010) {
                // write the previous group
                groups += groupSize
                groupWeights += prevWeight
              }
              groupSize = 1
              prevWeight = labeledPoint.weight
              prevGroup = labeledPoint.group
            } else {
              // for the same group
              if (prevWeight != labeledPoint.weight) {
                throw new IllegalArgumentException("the instances in the same group have to be" +
                  s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
              }
              groupSize = groupSize + 1
            }
          }
          labeledPoint
        }
        val dm = new DMatrix(transformedIter, null, missing)
        columnIndices.marginId.foreach(_ => dm.setBaseMargin(margins.result()))
        if (columnIndices.groupId.isDefined) {
          if (prevGroup != -101011) {
            // write the last group
            groups += groupSize
            groupWeights += prevWeight
          }
          dm.setGroup(groups.result())
          // The new DMatrix() will set the weights for each instance. But ranking requires
          // 1 weight for each group, so need to reset the weight.
          // This is definitely optimized by moving setting group/base margin into JNI.
          dm.setWeight(groupWeights.result())
        }
        dm
      } else {
        new DMatrix(iter, null, missing)
      }
      featureTypes.foreach(dmatrix.setFeatureTypes)
      featureNames.foreach(dmatrix.setFeatureNames)
      dmatrix
    }

    getEvalDataset().map { eval =>
      val (evalDf, _) = preprocess(eval)
      val evalRDD = toXGBLabeledPoint(evalDf, columnIndices)
      trainRDD.zipPartitions(evalRDD) { (left, right) =>
        val trainDMatrix = buildDMatrix(left)
        val evalDMatrix = buildDMatrix(right)
        val watches = new Watches(Array(trainDMatrix, evalDMatrix),
          Array(Utils.TRAIN_NAME, Utils.VALIDATION_NAME), None)
        Iterator.single(watches)
      }
    }.getOrElse(
      trainRDD.mapPartitions { iter =>
        val dm = buildDMatrix(iter)
        val watches = new Watches(Array(dm), Array(Utils.TRAIN_NAME), None)
        Iterator.single(watches)
      }
    )
  }

  protected def createModel(booster: Booster, summary: XGBoostTrainingSummary): M

  private[spark] def getRuntimeParameters(isLocal: Boolean): RuntimeParams = {
    val runOnGpu = if (getDevice != "cpu" || getTreeMethod == "gpu_hist") true else false
    RuntimeParams(
      getNumWorkers,
      getNumRound,
      TrackerConf(getRabitTrackerTimeout, getRabitTrackerHostIp, getRabitTrackerPort),
      getNumEarlyStoppingRounds,
      getDevice,
      isLocal,
      runOnGpu,
      Option(getCustomObj),
      Option(getCustomEval)
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
  protected[spark] def validate(dataset: Dataset[_]): Unit = {
    validateSparkSslConf(dataset.sparkSession)
    val schema = dataset.schema
    SparkUtils.checkNumericType(schema, $(labelCol))
    if (isDefinedNonEmpty(weightCol)) {
      SparkUtils.checkNumericType(schema, $(weightCol))
    }

    if (isDefinedNonEmpty(baseMarginCol)) {
      SparkUtils.checkNumericType(schema, $(baseMarginCol))
    }

    val taskCpus = dataset.sparkSession.sparkContext.getConf.getInt("spark.task.cpus", 1)
    if (isDefined(nthread)) {
      require(getNthread <= taskCpus,
        s"the nthread configuration ($getNthread) must be no larger than " +
          s"spark.task.cpus ($taskCpus)")
    } else {
      setNthread(taskCpus)
    }
  }

  def train(dataset: Dataset[_]): M = {
    validate(dataset)

    val rdd = if (isPluginEnabled(dataset)) {
      getPlugin.get.buildRddWatches(this, dataset)
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

  override def copy(extra: ParamMap): Learner = defaultCopy(extra).asInstanceOf[Learner]
}

/**
 * Indicate what to be predicted
 *
 * @param predLeaf    predicate leaf
 * @param predContrib predicate contribution
 * @param predRaw     predicate raw
 * @param predTmp     predicate probability for classification, and raw for regression
 */
private[spark] case class PredictedColumns(
    predLeaf: Boolean,
    predContrib: Boolean,
    predRaw: Boolean,
    predTmp: Boolean)

/**
 * XGBoost base model
 */
private[spark] trait XGBoostModel[M <: XGBoostModel[M]] extends Model[M] with MLWritable
  with XGBoostParams[M] with SparkParams[M] with ParamUtils[M] with PluginMixin {

  protected val TMP_TRANSFORMED_COL = "_tmp_xgb_transformed_col"

  override def copy(extra: ParamMap): M = defaultCopy(extra).asInstanceOf[M]

  /**
   * Get the native XGBoost Booster
   *
   * @return
   */
  def nativeBooster: Booster

  def summary: Option[XGBoostTrainingSummary]

  protected[spark] def postTransform(dataset: Dataset[_], pred: PredictedColumns): Dataset[_] = {
    var output = dataset
    // Convert leaf/contrib to the vector from array
    if (pred.predLeaf) {
      output = output.withColumn(getLeafPredictionCol,
        array_to_vector(output.col(getLeafPredictionCol)))
    }

    if (pred.predContrib) {
      output = output.withColumn(getContribPredictionCol,
        array_to_vector(output.col(getContribPredictionCol)))
    }
    output
  }

  /**
   * Preprocess the schema before transforming.
   *
   * @return the transformed schema and the
   */
  private[spark] def preprocess(dataset: Dataset[_]): (StructType, PredictedColumns) = {
    // Be careful about the order of columns
    var schema = dataset.schema

    /** If the parameter is defined, add it to schema and turn true */
    def addToSchema(param: Param[String], colName: Option[String] = None): Boolean = {
      if (isDefinedNonEmpty(param)) {
        val name = colName.getOrElse($(param))
        schema = schema.add(StructField(name, ArrayType(FloatType)))
        true
      } else {
        false
      }
    }

    val predLeaf = addToSchema(leafPredictionCol)
    val predContrib = addToSchema(contribPredictionCol)

    var predRaw = false
    // For classification case, the tranformed col is probability,
    // while for others, it's the prediction value.
    var predTmp = false
    this match {
      case p: XGBProbabilisticClassifierParams[_] => // classification case
        predRaw = addToSchema(p.rawPredictionCol)
        predTmp = addToSchema(p.probabilityCol, Some(TMP_TRANSFORMED_COL))

        if (isDefinedNonEmpty(predictionCol)) {
          // Let's use transformed col to calculate the prediction
          if (!predTmp) {
            // Add the transformed col for predition
            schema = schema.add(
              StructField(TMP_TRANSFORMED_COL, ArrayType(FloatType)))
            predTmp = true
          }
        }
      case _ =>
        // Rename TMP_TRANSFORMED_COL to prediction in the postTransform.
        predTmp = addToSchema(predictionCol, Some(TMP_TRANSFORMED_COL))
    }
    (schema, PredictedColumns(predLeaf, predContrib, predRaw, predTmp))
  }

  /** Predict */
  private[spark] def predictInternal(booster: Booster, dm: DMatrix, pred: PredictedColumns,
                                     batchRow: Iterator[Row]): Seq[Row] = {
    var tmpOut = batchRow.toSeq.map(_.toSeq)
    val zip = (left: Seq[Seq[_]], right: Array[Array[Float]]) => left.zip(right).map {
      case (a, b) => a ++ Seq(b)
    }
    if (pred.predLeaf) {
      tmpOut = zip(tmpOut, booster.predictLeaf(dm))
    }
    if (pred.predContrib) {
      tmpOut = zip(tmpOut, booster.predictContrib(dm))
    }
    if (pred.predRaw) {
      tmpOut = zip(tmpOut, booster.predict(dm, outPutMargin = true))
    }
    if (pred.predTmp) {
      tmpOut = zip(tmpOut, booster.predict(dm, outPutMargin = false))
    }
    tmpOut.map(Row.fromSeq)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    if (getPlugin.isDefined) {
      return getPlugin.get.transform(this, dataset)
    }

    val (schema, pred) = preprocess(dataset)
    val bBooster = dataset.sparkSession.sparkContext.broadcast(nativeBooster)
    // TODO configurable
    val inferBatchSize = 32 << 10
    // Broadcast the booster to each executor.
    val featureName = getFeaturesCol
    val missing = getMissing

    val output = dataset.toDF().mapPartitions { rowIter =>
      rowIter.grouped(inferBatchSize).flatMap { batchRow =>
        val features = batchRow.iterator.map(row => row.getAs[Vector](
          row.fieldIndex(featureName)))
        // DMatrix used to prediction
        val dm = new DMatrix(features.map(_.asXGB), null, missing)
        try {
          predictInternal(bBooster.value, dm, pred, batchRow.toIterator)
        } finally {
          dm.delete()
        }
      }

    }(Encoders.row(schema))
    bBooster.unpersist(blocking = false)
    postTransform(output, pred).toDF()
  }

  override def write: MLWriter = new XGBoostModelWriter(this)

  protected def predictSingleInstance(features: Vector): Array[Float] = {
    if (nativeBooster == null) {
      throw new IllegalArgumentException("The model has not been trained")
    }
    val dm = new DMatrix(Iterator(features.asXGB), null, getMissing)
    nativeBooster.predict(data = dm)(0)
  }
}

/**
 * Class to write the model
 *
 * @param instance model to be written
 */
private[spark] class XGBoostModelWriter(instance: XGBoostModel[_]) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    if (Option(instance.nativeBooster).isEmpty) {
      throw new RuntimeException("The XGBoost model has not been trained")
    }
    SparkUtils.saveMetadata(instance, path, sc)

    // Save model data
    val dataPath = new Path(path, "data").toString
    val internalPath = new Path(dataPath, "model")
    val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
    val format = optionMap.getOrElse("format", JBooster.DEFAULT_FORMAT)
    try {
      instance.nativeBooster.saveModel(outputStream, format)
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

// Trait for Ranker and Regressor Model
private[spark] trait RankerRegressorBaseModel[M <: XGBoostModel[M]] extends XGBoostModel[M] {

  override protected[spark] def postTransform(dataset: Dataset[_],
                                              pred: PredictedColumns): Dataset[_] = {
    var output = super.postTransform(dataset, pred)
    if (isDefinedNonEmpty(predictionCol) && pred.predTmp) {
      val predictUDF = udf { (originalPrediction: mutable.WrappedArray[Float]) =>
        originalPrediction(0).toDouble
      }
      output = output
        .withColumn($(predictionCol), predictUDF(col(TMP_TRANSFORMED_COL)))
        .drop(TMP_TRANSFORMED_COL)
    }
    output
  }

}
