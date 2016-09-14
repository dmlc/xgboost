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
import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Model, PredictionModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{VectorUDT, DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkContext, TaskContext}

class XGBoostModel(_booster: Booster) extends Model[XGBoostModel] with Serializable {

  var inputCol = "features"
  var outputCol = "prediction"
  var outputType: DataType = ArrayType(elementType = FloatType, containsNull = false)

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
  def eval(evalDataset: RDD[LabeledPoint], evalName: String, evalFunc: EvalTrait = null,
           iter: Int = -1, useExternalCache: Boolean = false): String = {
    require(evalFunc != null || iter != -1, "you have to specify value of either eval or iter")
    val broadcastBooster = evalDataset.sparkContext.broadcast(_booster)
    val appName = evalDataset.context.appName
    val allEvalMetrics = evalDataset.mapPartitions {
      labeledPointsPartition =>
        if (labeledPointsPartition.hasNext) {
          val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
          Rabit.init(rabitEnv.asJava)
          import DataUtils._
          val cacheFileName = {
            if (useExternalCache) {
              s"$appName-${TaskContext.get().stageId()}-deval_cache-${TaskContext.getPartitionId()}"
            } else {
              null
            }
          }
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
   * @param useExternalCache whether to use external cache for the test set
   */
  def predict(testSet: RDD[Vector], useExternalCache: Boolean = false): RDD[Array[Array[Float]]] = {
    import DataUtils._
    val broadcastBooster = testSet.sparkContext.broadcast(_booster)
    val appName = testSet.context.appName
    testSet.mapPartitions { testSamples =>
      if (testSamples.hasNext) {
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

  /**
   * Predict result with the given test set (represented as RDD)
   *
   * @param testSet test set represented as RDD
   * @param missingValue the specified value to represent the missing value
   */
  def predict(testSet: RDD[DenseVector], missingValue: Float): RDD[Array[Array[Float]]] = {
    val broadcastBooster = testSet.sparkContext.broadcast(_booster)
    testSet.mapPartitions { testSamples =>
      val sampleArray = testSamples.toList
      val numRows = sampleArray.size
      val numColumns = sampleArray.head.size
      if (numRows == 0) {
        Iterator()
      } else {
        // translate to required format
        val flatSampleArray = new Array[Float](numRows * numColumns)
        for (i <- flatSampleArray.indices) {
          flatSampleArray(i) = sampleArray(i / numColumns).values(i % numColumns).toFloat
        }
        val dMatrix = new DMatrix(flatSampleArray, numRows, numColumns, missingValue)
        Iterator(broadcastBooster.value.predict(dMatrix))
      }
    }
  }

  /**
   * Predict leaf instances with the given test set (represented as RDD)
   *
   * @param testSet test set represented as RDD
   */
  def predictLeaves(testSet: RDD[Vector]): RDD[Array[Array[Float]]] = {
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
   * Save the model as to HDFS-compatible file system.
   *
   * @param modelPath The model path as in Hadoop path.
   */
  def saveModelAsHadoopFile(modelPath: String)(implicit sc: SparkContext): Unit = {
    val path = new Path(modelPath)
    val outputStream = path.getFileSystem(sc.hadoopConfiguration).create(path)
    _booster.saveModel(outputStream)
    outputStream.close()
  }

  def booster: Booster = _booster

  override val uid: String = Identifiable.randomUID("XGBoostModel")

  override def copy(extra: ParamMap): XGBoostModel = {
    defaultCopy(extra)
  }

  /**
   * append leaf index of each row as an additional column in the original dataset
   *
   * @return the original dataframe with an additional column containing prediction results
   */
  def transformLeaf(testSet: Dataset[_]): Unit = {
    outputCol = "predLeaf"
    transformSchema(testSet.schema, logging = true)
    val broadcastBooster = testSet.sparkSession.sparkContext.broadcast(_booster)
    val instances = testSet.rdd.mapPartitions {
      rowIterator =>
        if (rowIterator.hasNext) {
          val (rowItr1, rowItr2) = rowIterator.duplicate
          val vectorIterator = rowItr2.map(row => row.asInstanceOf[Row].getAs[Vector](inputCol)).
            toList.iterator
          import DataUtils._
          val testDataset = new DMatrix(vectorIterator, null)
          val rowPredictResults = broadcastBooster.value.predictLeaf(testDataset)
          val predictResults = rowPredictResults.map(prediction => Row(prediction)).iterator
          rowItr1.zip(predictResults).map {
            case (originalColumns: Row, predictColumn: Row) =>
              Row.fromSeq(originalColumns.toSeq ++ predictColumn.toSeq)
          }
        } else {
          Iterator[Row]()
        }
    }
    testSet.sparkSession.createDataFrame(instances, testSet.schema.add(outputCol, outputType)).
      cache()
  }

  /**
   * produces the prediction results and append as an additional column in the original dataset
   * NOTE: the prediction results is kept as the original format of xgboost
   *
   * @return the original dataframe with an additional column containing prediction results
   */
  override def transform(testSet: Dataset[_]): DataFrame = {
    transform(testSet, None)
  }

  /**
   * produces the prediction results and append as an additional column in the original dataset
   * NOTE: the prediction results is transformed by applying the transformation function
   * predictResultTrans to the original xgboost output
   *
   * @param rawPredictTransformer the function to transform xgboost output to the expected format
   * @return the original dataframe with an additional column containing prediction results
   */
  def transform(testSet: Dataset[_], rawPredictTransformer: Option[Array[Float] => DataType]):
      DataFrame = {
    transformSchema(testSet.schema, logging = true)
    val broadcastBooster = testSet.sparkSession.sparkContext.broadcast(_booster)
    val instances = testSet.rdd.mapPartitions {
      rowIterator =>
        if (rowIterator.hasNext) {
          val (rowItr1, rowItr2) = rowIterator.duplicate
          val vectorIterator = rowItr2.map(row => row.asInstanceOf[Row].getAs[Vector](inputCol)).
            toList.iterator
          import DataUtils._
          val testDataset = new DMatrix(vectorIterator, null)
          val rowPredictResults = broadcastBooster.value.predict(testDataset)
          val predictResults = {
            if (rawPredictTransformer.isDefined) {
              rowPredictResults.map(prediction =>
                Row(rawPredictTransformer.get(prediction))).iterator
            } else {
              rowPredictResults.map(prediction => Row(prediction)).iterator
            }
          }
          rowItr1.zip(predictResults).map {
            case (originalColumns: Row, predictColumn: Row) =>
              Row.fromSeq(originalColumns.toSeq ++ predictColumn.toSeq)
          }
        } else {
          Iterator[Row]()
        }
    }
    testSet.sparkSession.createDataFrame(instances, testSet.schema.add(outputCol, outputType)).
      cache()
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains(outputCol)) {
      throw new IllegalArgumentException(s"Output column $outputCol already exists.")
    }
    val inputType = schema(inputCol).dataType
    require(inputType.equals(new VectorUDT),
      s"the type of input column $inputCol has to be VectorUDT")
    val outputFields = schema.fields :+ StructField(outputCol, outputType, nullable = false)
    StructType(outputFields)
  }
}
