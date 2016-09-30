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

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix, Rabit}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait}
import org.apache.hadoop.fs.{FSDataOutputStream, Path}
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector, Vector => MLVector}
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{FloatType, ArrayType, DataType}
import org.apache.spark.{SparkContext, TaskContext}

abstract class XGBoostModel(_booster: Booster)
  extends PredictionModel[MLVector, XGBoostModel] with Serializable with Params {

  def setLabelCol(name: String): XGBoostModel = set(labelCol, name)

  // scalastyle:off

  final val useExternalMemory: Param[Boolean] = new Param[Boolean](this, "useExternalMemory", "whether to use external memory for prediction")

  setDefault(useExternalMemory, false)

  def setExternalMemory(value: Boolean): XGBoostModel = set(useExternalMemory, value)

  // scalastyle:on

  /**
   * Predict leaf instances with the given test set (represented as RDD)
   *
   * @param testSet test set represented as RDD
   */
  def predictLeaves(testSet: RDD[MLVector]): RDD[Array[Array[Float]]] = {
    import DataUtils._
    val broadcastBooster = testSet.sparkContext.broadcast(_booster)
    testSet.mapPartitions { testSamples =>
      if (testSamples.hasNext) {
        val dMatrix = new DMatrix(new JDMatrix(testSamples, null))
        Iterator(broadcastBooster.value.predictLeaf(dMatrix))
      } else {
        Iterator()
      }
    }
  }

  /**
   * evaluate XGBoostModel with a RDD-wrapped dataset
   *
   * NOTE: you have to specify value of either eval or iter; when you specify both, this method
   * adopts the default eval metric of model
   *
   * @param evalDataset the dataset used for evaluation
   * @param evalName the name of evaluation
   * @param evalFunc the customized evaluation function, null by default to use the default metric
   *             of model
   * @param iter the current iteration, -1 to be null to use customized evaluation functions
   * @param useExternalCache if use external cache
   * @return the average metric over all partitions
   */
  def eval(evalDataset: RDD[MLLabeledPoint], evalName: String, evalFunc: EvalTrait = null,
           iter: Int = -1, useExternalCache: Boolean = false): String = {
    require(evalFunc != null || iter != -1, "you have to specify the value of either eval or iter")
    val broadcastBooster = evalDataset.sparkContext.broadcast(_booster)
    val appName = evalDataset.context.appName
    val allEvalMetrics = evalDataset.mapPartitions {
      labeledPointsPartition =>
        if (labeledPointsPartition.hasNext) {
          val rabitEnv = Map("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString)
          Rabit.init(rabitEnv.asJava)
          val cacheFileName = {
            if (useExternalCache) {
              s"$appName-${TaskContext.get().stageId()}-$evalName" +
                s"-deval_cache-${TaskContext.getPartitionId()}"
            } else {
              null
            }
          }
          import DataUtils._
          val dMatrix = new DMatrix(labeledPointsPartition, cacheFileName)
          if (iter == -1) {
            val predictions = broadcastBooster.value.predict(dMatrix)
            Rabit.shutdown()
            Iterator(Some((evalName, evalFunc.eval(predictions, dMatrix))))
          } else {
            val predStr = broadcastBooster.value.evalSet(Array(dMatrix), Array(evalName), iter)
            val Array(evName, predNumeric) = predStr.split(":")
            Rabit.shutdown()
            Iterator(Some(evName, predNumeric.toFloat))
          }
        } else {
          Iterator(None)
        }
    }.filter(_.isDefined).collect()
    val evalPrefix = allEvalMetrics.map(_.get._1).head
    val evalMetricMean = allEvalMetrics.map(_.get._2).sum / allEvalMetrics.length
    s"$evalPrefix = $evalMetricMean"
  }

  /**
   * Predict result with the given test set (represented as RDD)
   *
   * @param testSet test set represented as RDD
   * @param missingValue the specified value to represent the missing value
   */
  def predict(testSet: RDD[MLDenseVector], missingValue: Float): RDD[Array[Array[Float]]] = {
    val broadcastBooster = testSet.sparkContext.broadcast(_booster)
    testSet.mapPartitions { testSamples =>
      val sampleArray = testSamples.toList
      val numRows = sampleArray.size
      val numColumns = sampleArray.head.size
      if (numRows == 0) {
        Iterator()
      } else {
        val rabitEnv = Map("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString)
        Rabit.init(rabitEnv.asJava)
        // translate to required format
        val flatSampleArray = new Array[Float](numRows * numColumns)
        for (i <- flatSampleArray.indices) {
          flatSampleArray(i) = sampleArray(i / numColumns).values(i % numColumns).toFloat
        }
        val dMatrix = new DMatrix(flatSampleArray, numRows, numColumns, missingValue)
        Rabit.shutdown()
        Iterator(broadcastBooster.value.predict(dMatrix))
      }
    }
  }

  /**
   * Predict result with the given test set (represented as RDD)
   *
   * @param testSet test set represented as RDD
   * @param useExternalCache whether to use external cache for the test set
   */
  def predict(testSet: RDD[MLVector], useExternalCache: Boolean = false):
      RDD[Array[Array[Float]]] = {
    val broadcastBooster = testSet.sparkContext.broadcast(_booster)
    val appName = testSet.context.appName
    testSet.mapPartitions { testSamples =>
      if (testSamples.hasNext) {
        import DataUtils._
        val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
        Rabit.init(rabitEnv.asJava)
        val cacheFileName = {
          if (useExternalCache) {
            s"$appName-${TaskContext.get().stageId()}-dtest_cache-${TaskContext.getPartitionId()}"
          } else {
            null
          }
        }
        val dMatrix = new DMatrix(new JDMatrix(testSamples, cacheFileName))
        val res = broadcastBooster.value.predict(dMatrix)
        Rabit.shutdown()
        Iterator(res)
      } else {
        Iterator()
      }
    }
  }

  protected def transformImpl(testSet: Dataset[_]): DataFrame

  /**
   * append leaf index of each row as an additional column in the original dataset
   *
   * @return the original dataframe with an additional column containing prediction results
   */
  def transformLeaf(testSet: Dataset[_]): DataFrame = {
    val predictRDD = produceRowRDD(testSet, predLeaf = true)
    setPredictionCol("predLeaf")
    transformSchema(testSet.schema, logging = true)
    testSet.sparkSession.createDataFrame(predictRDD, testSet.schema.add($(predictionCol),
      ArrayType(FloatType, containsNull = false)))
  }

  protected def produceRowRDD(testSet: Dataset[_], outputMargin: Boolean = false,
      predLeaf: Boolean = false): RDD[Row] = {
    val broadcastBooster = testSet.sparkSession.sparkContext.broadcast(_booster)
    val appName = testSet.sparkSession.sparkContext.appName
    testSet.rdd.mapPartitions {
      rowIterator =>
        if (rowIterator.hasNext) {
          val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
          Rabit.init(rabitEnv.asJava)
          val (rowItr1, rowItr2) = rowIterator.duplicate
          val vectorIterator = rowItr2.map(row => row.asInstanceOf[Row].getAs[MLVector](
            $(featuresCol))).toList.iterator
          import DataUtils._
          val cachePrefix = {
            if ($(useExternalMemory)) {
              s"$appName-${TaskContext.get().stageId()}-dtest_cache-${TaskContext.getPartitionId()}"
            } else {
              null
            }
          }
          val testDataset = new DMatrix(vectorIterator, cachePrefix)
          val rawPredictResults = {
            if (!predLeaf) {
              broadcastBooster.value.predict(testDataset, outputMargin).
                map(Row(_)).iterator
            } else {
              broadcastBooster.value.predictLeaf(testDataset).map(Row(_)).iterator
            }
          }
          Rabit.shutdown()
          // concatenate original data partition and predictions
          rowItr1.zip(rawPredictResults).map {
            case (originalColumns: Row, predictColumn: Row) =>
              Row.fromSeq(originalColumns.toSeq ++ predictColumn.toSeq)
          }
        } else {
          Iterator[Row]()
        }
    }
  }

  /**
   * produces the prediction results and append as an additional column in the original dataset
   * NOTE: the prediction results is kept as the original format of xgboost
   *
   * @return the original dataframe with an additional column containing prediction results
   */
  override def transform(testSet: Dataset[_]): DataFrame = {
    transformImpl(testSet)
  }

  private def saveGeneralModelParam(outputStream: FSDataOutputStream): Unit = {
    outputStream.writeUTF(getFeaturesCol)
    outputStream.writeUTF(getLabelCol)
    outputStream.writeUTF(getPredictionCol)
  }

  /**
   * Save the model as to HDFS-compatible file system.
   *
   * @param modelPath The model path as in Hadoop path.
   */
  def saveModelAsHadoopFile(modelPath: String)(implicit sc: SparkContext): Unit = {
    val path = new Path(modelPath)
    val outputStream = path.getFileSystem(sc.hadoopConfiguration).create(path)
    // output model type
    this match {
      case model: XGBoostClassificationModel =>
        outputStream.writeUTF("_cls_")
        saveGeneralModelParam(outputStream)
        outputStream.writeUTF(model.getRawPredictionCol)
        // threshold
        // threshold length
        if (!isDefined(model.thresholds)) {
          outputStream.writeInt(-1)
        } else {
          val thresholdLength = model.getThresholds.length
          outputStream.writeInt(thresholdLength)
          for (i <- 0 until thresholdLength) {
            outputStream.writeDouble(model.getThresholds(i))
          }
        }
      case model: XGBoostRegressionModel =>
        outputStream.writeUTF("_reg_")
        // eventual prediction col
        saveGeneralModelParam(outputStream)
    }
    // booster
    _booster.saveModel(outputStream)
    outputStream.close()
  }

  // override protected def featuresDataType: DataType = new VectorUDT

  def booster: Booster = _booster
}
