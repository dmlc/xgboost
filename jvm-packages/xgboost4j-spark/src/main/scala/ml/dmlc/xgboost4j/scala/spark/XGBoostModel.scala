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

import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.{TaskContext, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, Booster}

class XGBoostModel(_booster: Booster) extends Serializable {

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
        val cacheFileName = {
          if (useExternalCache) {
            s"$appName-dtest_cache-${TaskContext.getPartitionId()}"
          } else {
            null
          }
        }
        val dMatrix = new DMatrix(new JDMatrix(testSamples, cacheFileName))
        Iterator(broadcastBooster.value.predict(dMatrix))
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
   * Predict result with the given test set (represented as DMatrix)
   *
   * @param testSet test set represented as DMatrix
   */
  def predict(testSet: DMatrix): Array[Array[Float]] = {
    _booster.predict(testSet, true, 0)
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
        Iterator(broadcastBooster.value.predictLeaf(dMatrix, 0))
      } else {
        Iterator()
      }
    }
  }

  /**
   * Predict leaf instances with the given test set (represented as DMatrix)
   *
   * @param testSet test set represented as DMatrix
   */
  def predictLeaves(testSet: DMatrix): Array[Array[Float]] = {
    _booster.predictLeaf(testSet, 0)
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

  /**
   * Get the booster instance of this model
   */
  def booster: Booster = _booster
}
