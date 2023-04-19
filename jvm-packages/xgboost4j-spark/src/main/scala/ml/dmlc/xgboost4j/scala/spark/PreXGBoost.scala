/*
 Copyright (c) 2021-2023 by Contributors

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

import java.nio.file.Files
import java.util.ServiceLoader

import scala.collection.JavaConverters._
import scala.collection.{AbstractIterator, Iterator, mutable}

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.scala.spark.util.DataUtils.PackedParams
import ml.dmlc.xgboost4j.scala.spark.params.XGBoostEstimatorCommon
import ml.dmlc.xgboost4j.scala.spark.util.DataUtils

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.logging.LogFactory

import org.apache.spark.TaskContext
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

/**
 * PreXGBoost serves preparing data before training and transform
 */
object PreXGBoost extends PreXGBoostProvider {

  private val logger = LogFactory.getLog("XGBoostSpark")

  private lazy val defaultBaseMarginColumn = lit(Float.NaN)
  private lazy val defaultWeightColumn = lit(1.0)
  private lazy val defaultGroupColumn = lit(-1)

  // Find the correct PreXGBoostProvider by ServiceLoader
  private val optionProvider: Option[PreXGBoostProvider] = {
    val classLoader = Option(Thread.currentThread().getContextClassLoader)
      .getOrElse(getClass.getClassLoader)

    val serviceLoader = ServiceLoader.load(classOf[PreXGBoostProvider], classLoader)

    // For now, we only trust GpuPreXGBoost.
    serviceLoader.asScala.filter(x => x.getClass.getName.equals(
      "ml.dmlc.xgboost4j.scala.rapids.spark.GpuPreXGBoost")).toList match {
      case Nil => None
      case head::Nil =>
        Some(head)
      case _ => None
    }
  }

  /**
   * Transform schema
   *
   * @param xgboostEstimator supporting XGBoostClassifier/XGBoostClassificationModel and
   *                 XGBoostRegressor/XGBoostRegressionModel
   * @param schema   the input schema
   * @return the transformed schema
   */
  override def transformSchema(
      xgboostEstimator: XGBoostEstimatorCommon,
      schema: StructType): StructType = {

    if (optionProvider.isDefined && optionProvider.get.providerEnabled(None)) {
      return optionProvider.get.transformSchema(xgboostEstimator, schema)
    }

    xgboostEstimator match {
      case est: XGBoostClassifier => est.transformSchemaInternal(schema)
      case model: XGBoostClassificationModel => model.transformSchemaInternal(schema)
      case reg: XGBoostRegressor => reg.transformSchemaInternal(schema)
      case model: XGBoostRegressionModel => model.transformSchemaInternal(schema)
      case _ => throw new RuntimeException("Unsupporting " + xgboostEstimator)
    }
  }

  /**
   * Convert the Dataset[_] to RDD[() => Watches] which will be fed to XGBoost
   *
   * @param estimator supports XGBoostClassifier and XGBoostRegressor
   * @param dataset the training data
   * @param params all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[() => Watches]], Option[ RDD[_] ])
   *         RDD[() => Watches] will be used as the training input
   *         Option[RDD[_]\] is the optional cached RDD
   */
  override def buildDatasetToRDD(
      estimator: Estimator[_],
      dataset: Dataset[_],
      params: Map[String, Any]): XGBoostExecutionParams =>
    (RDD[() => Watches], Option[RDD[_]]) = {

    if (optionProvider.isDefined && optionProvider.get.providerEnabled(Some(dataset))) {
      return optionProvider.get.buildDatasetToRDD(estimator, dataset, params)
    }

    val (packedParams, evalSet, xgbInput) = estimator match {
      case est: XGBoostEstimatorCommon =>
        // get weight column, if weight is not defined, default to lit(1.0)
        val weight = if (!est.isDefined(est.weightCol) || est.getWeightCol.isEmpty) {
          defaultWeightColumn
        } else col(est.getWeightCol)

        // get base-margin column, if base-margin is not defined, default to lit(Float.NaN)
        val baseMargin = if (!est.isDefined(est.baseMarginCol) || est.getBaseMarginCol.isEmpty) {
            defaultBaseMarginColumn
          } else col(est.getBaseMarginCol)

        val group = est match {
          case regressor: XGBoostRegressor =>
            // get group column, if group is not defined, default to lit(-1)
            Some(
              if (!regressor.isDefined(regressor.groupCol) || regressor.getGroupCol.isEmpty) {
                defaultGroupColumn
              } else col(regressor.getGroupCol)
            )
          case _ => None

        }

        val (xgbInput, featuresName) = est.vectorize(dataset)

        val evalSets = est.getEvalSets(params).transform((_, df) => {
          val (dfTransformed, _) = est.vectorize(df)
          dfTransformed
        })

        (PackedParams(col(est.getLabelCol), col(featuresName), weight, baseMargin, group,
          est.getNumWorkers, est.needDeterministicRepartitioning), evalSets, xgbInput)

      case _ => throw new RuntimeException("Unsupporting " + estimator)
    }

    // transform the training Dataset[_] to RDD[XGBLabeledPoint]
    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      packedParams, xgbInput.asInstanceOf[DataFrame]).head

    // transform the eval Dataset[_] to RDD[XGBLabeledPoint]
    val evalRDDMap = evalSet.map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(packedParams,
          dataFrame.asInstanceOf[DataFrame]).head)
    }

    val hasGroup = packedParams.group.map(_ != defaultGroupColumn).getOrElse(false)

    xgbExecParams: XGBoostExecutionParams =>
      composeInputData(trainingSet, hasGroup, packedParams.numWorkers) match {
        case Left(trainingData) =>
          val cachedRDD = if (xgbExecParams.cacheTrainingSet) {
            Some(trainingData.persist(StorageLevel.MEMORY_AND_DISK))
          } else None
          (trainForRanking(trainingData, xgbExecParams, evalRDDMap), cachedRDD)
        case Right(trainingData) =>
          val cachedRDD = if (xgbExecParams.cacheTrainingSet) {
            Some(trainingData.persist(StorageLevel.MEMORY_AND_DISK))
          } else None
          (trainForNonRanking(trainingData, xgbExecParams, evalRDDMap), cachedRDD)
      }

  }

  /**
   * Transform Dataset
   *
   * @param model supporting [[XGBoostClassificationModel]] and [[XGBoostRegressionModel]]
   * @param dataset the input Dataset to transform
   * @return the transformed DataFrame
   */
  override def transformDataset(model: Model[_], dataset: Dataset[_]): DataFrame = {

    if (optionProvider.isDefined && optionProvider.get.providerEnabled(Some(dataset))) {
      return optionProvider.get.transformDataset(model, dataset)
    }

    /** get the necessary parameters */
    val (booster, inferBatchSize, xgbInput, featuresCol, useExternalMemory, missing,
    allowNonZeroForMissing, predictFunc, schema) =
      model match {
        case m: XGBoostClassificationModel =>
          val (xgbInput, featuresName) = m.vectorize(dataset)
          // predict and turn to Row
          val predictFunc =
            (booster: Booster, dm: DMatrix, originalRowItr: Iterator[Row]) => {
              val Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr) =
                m.producePredictionItrs(booster, dm)
              m.produceResultIterator(originalRowItr, rawPredictionItr, probabilityItr,
                predLeafItr, predContribItr)
            }

          // prepare the final Schema
          var schema = StructType(xgbInput.schema.fields ++
            Seq(StructField(name = XGBoostClassificationModel._rawPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false)) ++
            Seq(StructField(name = XGBoostClassificationModel._probabilityCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false)))

          if (m.isDefined(m.leafPredictionCol)) {
            schema = schema.add(StructField(name = m.getLeafPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false))
          }
          if (m.isDefined(m.contribPredictionCol)) {
            schema = schema.add(StructField(name = m.getContribPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false))
          }

          (m._booster, m.getInferBatchSize, xgbInput, featuresName, m.getUseExternalMemory,
            m.getMissing, m.getAllowNonZeroForMissingValue, predictFunc, schema)

        case m: XGBoostRegressionModel =>
          // predict and turn to Row
          val (xgbInput, featuresName) = m.vectorize(dataset)
          val predictFunc =
            (booster: Booster, dm: DMatrix, originalRowItr: Iterator[Row]) => {
              val Array(rawPredictionItr, predLeafItr, predContribItr) =
                m.producePredictionItrs(booster, dm)
              m.produceResultIterator(originalRowItr, rawPredictionItr, predLeafItr, predContribItr)
            }

          // prepare the final Schema
          var schema = StructType(xgbInput.schema.fields ++
            Seq(StructField(name = XGBoostRegressionModel._originalPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false)))

          if (m.isDefined(m.leafPredictionCol)) {
            schema = schema.add(StructField(name = m.getLeafPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false))
          }
          if (m.isDefined(m.contribPredictionCol)) {
            schema = schema.add(StructField(name = m.getContribPredictionCol, dataType =
              ArrayType(FloatType, containsNull = false), nullable = false))
          }

          (m._booster, m.getInferBatchSize, xgbInput, featuresName, m.getUseExternalMemory,
            m.getMissing, m.getAllowNonZeroForMissingValue, predictFunc, schema)
      }

    val bBooster = xgbInput.sparkSession.sparkContext.broadcast(booster)
    val appName = xgbInput.sparkSession.sparkContext.appName

    val resultRDD = xgbInput.asInstanceOf[Dataset[Row]].rdd.mapPartitions { rowIterator =>
      new AbstractIterator[Row] {
        private var batchCnt = 0

        private val batchIterImpl = rowIterator.grouped(inferBatchSize).flatMap { batchRow =>
          val features = batchRow.iterator.map(row => row.getAs[Vector](featuresCol))

          import ml.dmlc.xgboost4j.scala.spark.util.DataUtils._
          val cacheInfo = {
            if (useExternalMemory) {
              s"$appName-${TaskContext.get().stageId()}-dtest_cache-" +
                s"${TaskContext.getPartitionId()}-batch-$batchCnt"
            } else {
              null
            }
          }

          val dm = new DMatrix(
            processMissingValues(features.map(_.asXGB), missing, allowNonZeroForMissing),
            cacheInfo)

          try {
            predictFunc(bBooster.value, dm, batchRow.iterator)
          } finally {
            batchCnt += 1
            dm.delete()
          }
        }

        override def hasNext: Boolean = batchIterImpl.hasNext

        override def next(): Row = batchIterImpl.next()

      }
    }

    bBooster.unpersist(blocking = false)
    xgbInput.sparkSession.createDataFrame(resultRDD, schema)
  }


  /**
   * Converting the RDD[XGBLabeledPoint] to the function to build RDD[() => Watches]
   *
   * @param trainingSet the input training RDD[XGBLabeledPoint]
   * @param evalRDDMap the eval set
   * @param hasGroup if has group
   * @return function to build (RDD[() => Watches], the cached RDD)
   */
  private[spark] def buildRDDLabeledPointToRDDWatches(
      trainingSet: RDD[XGBLabeledPoint],
      evalRDDMap: Map[String, RDD[XGBLabeledPoint]] = Map(),
      hasGroup: Boolean = false):
  XGBoostExecutionParams => (RDD[() => Watches], Option[RDD[_]]) = {

    xgbExecParams: XGBoostExecutionParams =>
      composeInputData(trainingSet, hasGroup, xgbExecParams.numWorkers) match {
        case Left(trainingData) =>
          val cachedRDD = if (xgbExecParams.cacheTrainingSet) {
            Some(trainingData.persist(StorageLevel.MEMORY_AND_DISK))
          } else None
          (trainForRanking(trainingData, xgbExecParams, evalRDDMap), cachedRDD)
        case Right(trainingData) =>
          val cachedRDD = if (xgbExecParams.cacheTrainingSet) {
            Some(trainingData.persist(StorageLevel.MEMORY_AND_DISK))
          } else None
          (trainForNonRanking(trainingData, xgbExecParams, evalRDDMap), cachedRDD)
      }
  }

  /**
   * Transform RDD according to group column
   *
   * @param trainingData the input XGBLabeledPoint RDD
   * @param hasGroup if has group column
   * @param nWorkers total xgboost number workers to run xgboost tasks
   * @return Either: the left is RDD with group, and the right is RDD without group
   */
  private def composeInputData(
      trainingData: RDD[XGBLabeledPoint],
      hasGroup: Boolean,
      nWorkers: Int): Either[RDD[Array[XGBLabeledPoint]], RDD[XGBLabeledPoint]] = {
    if (hasGroup) {
      Left(repartitionForTrainingGroup(trainingData, nWorkers))
    } else {
      Right(trainingData)
    }
  }

  /**
   * Repartition trainingData with group directly may cause data chaos, since the same group data
   * may be split into different partitions.
   *
   * The first step is to aggregate the same group into same partition
   * The second step is to repartition to nWorkers
   *
   * TODO, Could we repartition trainingData on group?
   */
  private[spark] def repartitionForTrainingGroup(trainingData: RDD[XGBLabeledPoint],
      nWorkers: Int): RDD[Array[XGBLabeledPoint]] = {
    val allGroups = aggByGroupInfo(trainingData)
    logger.info(s"repartitioning training group set to $nWorkers partitions")
    allGroups.repartition(nWorkers)
  }

  /**
   * Build RDD[() => Watches] for Ranking
   * @param trainingData the training data RDD
   * @param xgbExecutionParams xgboost execution params
   * @param evalSetsMap the eval RDD
   * @return RDD[() => Watches]
   */
  private def trainForRanking(
      trainingData: RDD[Array[XGBLabeledPoint]],
      xgbExecutionParam: XGBoostExecutionParams,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[() => Watches] = {
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions(labeledPointGroups => {
        val buildWatches = () => Watches.buildWatchesWithGroup(xgbExecutionParam,
          DataUtils.processMissingValuesWithGroup(labeledPointGroups, xgbExecutionParam.missing,
            xgbExecutionParam.allowNonZeroForMissing),
          getCacheDirName(xgbExecutionParam.useExternalMemory))
        Iterator.single(buildWatches)
      }).cache()
    } else {
      coPartitionGroupSets(trainingData, evalSetsMap, xgbExecutionParam.numWorkers).mapPartitions(
        labeledPointGroupSets => {
          val buildWatches = () => Watches.buildWatchesWithGroup(
            labeledPointGroupSets.map {
              case (name, iter) => (name, DataUtils.processMissingValuesWithGroup(iter,
                xgbExecutionParam.missing, xgbExecutionParam.allowNonZeroForMissing))
            },
            getCacheDirName(xgbExecutionParam.useExternalMemory))
          Iterator.single(buildWatches)
        }).cache()
    }
  }

  private def coPartitionGroupSets(
      aggedTrainingSet: RDD[Array[XGBLabeledPoint]],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int): RDD[(String, Iterator[Array[XGBLabeledPoint]])] = {
    val repartitionedDatasets = Map("train" -> aggedTrainingSet) ++ evalSets.map {
      case (name, rdd) => {
        val aggedRdd = aggByGroupInfo(rdd)
        if (aggedRdd.getNumPartitions != nWorkers) {
          name -> aggedRdd.repartition(nWorkers)
        } else {
          name -> aggedRdd
        }
      }
    }
    repartitionedDatasets.foldLeft(aggedTrainingSet.sparkContext.parallelize(
      Array.fill[(String, Iterator[Array[XGBLabeledPoint]])](nWorkers)(null), nWorkers)) {
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter) {
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  private def aggByGroupInfo(trainingData: RDD[XGBLabeledPoint]) = {
    val normalGroups: RDD[Array[XGBLabeledPoint]] = trainingData.mapPartitions(
      // LabeledPointGroupIterator returns (Boolean, Array[XGBLabeledPoint])
      new LabeledPointGroupIterator(_)).filter(!_.isEdgeGroup).map(_.points)

    // edge groups with partition id.
    val edgeGroups: RDD[(Int, XGBLabeledPointGroup)] = trainingData.mapPartitions(
      new LabeledPointGroupIterator(_)).filter(_.isEdgeGroup).map(
      group => (TaskContext.getPartitionId(), group))

    // group chunks from different partitions together by group id in XGBLabeledPoint.
    // use groupBy instead of aggregateBy since all groups within a partition have unique group ids.
    val stitchedGroups: RDD[Array[XGBLabeledPoint]] = edgeGroups.groupBy(_._2.groupId).map(
      groups => {
        val it: Iterable[(Int, XGBLabeledPointGroup)] = groups._2
        // sorted by partition id and merge list of Array[XGBLabeledPoint] into one array
        it.toArray.sortBy(_._1).flatMap(_._2.points)
      })
    normalGroups.union(stitchedGroups)
  }

  /**
   * Build RDD[() => Watches] for Non-Ranking
   * @param trainingData the training data RDD
   * @param xgbExecutionParams xgboost execution params
   * @param evalSetsMap the eval RDD
   * @return RDD[() => Watches]
   */
  private def trainForNonRanking(
      trainingData: RDD[XGBLabeledPoint],
      xgbExecutionParams: XGBoostExecutionParams,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[() => Watches] = {
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions { labeledPoints => {
        val buildWatches = () => Watches.buildWatches(xgbExecutionParams,
          DataUtils.processMissingValues(labeledPoints, xgbExecutionParams.missing,
            xgbExecutionParams.allowNonZeroForMissing),
          getCacheDirName(xgbExecutionParams.useExternalMemory))
        Iterator.single(buildWatches)
      }}.cache()
    } else {
      coPartitionNoGroupSets(trainingData, evalSetsMap, xgbExecutionParams.numWorkers).
        mapPartitions {
          nameAndLabeledPointSets =>
            val buildWatches = () => Watches.buildWatches(
              nameAndLabeledPointSets.map {
                case (name, iter) => (name, DataUtils.processMissingValues(iter,
                  xgbExecutionParams.missing, xgbExecutionParams.allowNonZeroForMissing))
              },
              getCacheDirName(xgbExecutionParams.useExternalMemory))
            Iterator.single(buildWatches)
        }.cache()
    }
  }

  private def coPartitionNoGroupSets(
      trainingData: RDD[XGBLabeledPoint],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int) = {
    // eval_sets is supposed to be set by the caller of [[trainDistributed]]
    val allDatasets = Map("train" -> trainingData) ++ evalSets
    val repartitionedDatasets = allDatasets.map { case (name, rdd) =>
      if (rdd.getNumPartitions != nWorkers) {
        (name, rdd.repartition(nWorkers))
      } else {
        (name, rdd)
      }
    }
    repartitionedDatasets.foldLeft(trainingData.sparkContext.parallelize(
      Array.fill[(String, Iterator[XGBLabeledPoint])](nWorkers)(null), nWorkers)) {
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter) {
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  private[scala] def getCacheDirName(useExternalMemory: Boolean): Option[String] = {
    val taskId = TaskContext.getPartitionId().toString
    if (useExternalMemory) {
      val dir = Files.createTempDirectory(s"${TaskContext.get().stageId()}-cache-$taskId")
      Some(dir.toAbsolutePath.toString)
    } else {
      None
    }
  }

}

class IteratorWrapper[T](arrayOfXGBLabeledPoints: Array[(String, Iterator[T])])
    extends Iterator[(String, Iterator[T])] {

  private var currentIndex = 0

  override def hasNext: Boolean = currentIndex <= arrayOfXGBLabeledPoints.length - 1

  override def next(): (String, Iterator[T]) = {
    currentIndex += 1
    arrayOfXGBLabeledPoints(currentIndex - 1)
  }
}

/**
 * Training data group in a RDD partition.
 *
 * @param groupId The group id
 * @param points Array of XGBLabeledPoint within the same group.
 * @param isEdgeGroup whether it is a first or last group in a RDD partition.
 */
private[spark] case class XGBLabeledPointGroup(
  groupId: Int,
  points: Array[XGBLabeledPoint],
  isEdgeGroup: Boolean)

/**
 * Within each RDD partition, group the <code>XGBLabeledPoint</code> by group id.</p>
 * And the first and the last groups may not have all the items due to the data partition.
 * <code>LabeledPointGroupIterator</code> organizes data in a tuple format:
 * (isFistGroup || isLastGroup, Array[XGBLabeledPoint]).</p>
 * The edge groups across partitions can be stitched together later.
 * @param base collection of <code>XGBLabeledPoint</code>
 */
private[spark] class LabeledPointGroupIterator(base: Iterator[XGBLabeledPoint])
  extends AbstractIterator[XGBLabeledPointGroup] {

  private var firstPointOfNextGroup: XGBLabeledPoint = null
  private var isNewGroup = false

  override def hasNext: Boolean = {
    base.hasNext || isNewGroup
  }

  override def next(): XGBLabeledPointGroup = {
    val builder = mutable.ArrayBuilder.make[XGBLabeledPoint]
    var isFirstGroup = true
    if (firstPointOfNextGroup != null) {
      builder += firstPointOfNextGroup
      isFirstGroup = false
    }

    isNewGroup = false
    while (!isNewGroup && base.hasNext) {
      val point = base.next()
      val groupId = if (firstPointOfNextGroup != null) firstPointOfNextGroup.group else point.group
      firstPointOfNextGroup = point
      if (point.group == groupId) {
        // add to current group
        builder += point
      } else {
        // start a new group
        isNewGroup = true
      }
    }

    val isLastGroup = !isNewGroup
    val result = builder.result()
    val group = XGBLabeledPointGroup(result(0).group, result, isFirstGroup || isLastGroup)

    group
  }
}
