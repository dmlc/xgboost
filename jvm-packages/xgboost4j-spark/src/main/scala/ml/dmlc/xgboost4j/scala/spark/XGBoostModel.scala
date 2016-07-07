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

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import ml.dmlc.xgboost4j.java.{Rabit, DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import scala.collection.JavaConverters._

class XGBoostModel(_booster: Booster)(implicit val sc: SparkContext) extends Serializable {

  /**
   * Predict result with the given testset (represented as RDD)
   * @param testSet test set representd as RDD
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
            s"$appName-dtest_cache-${TaskContext.getPartitionId()}"
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
   * predict result given the test data (represented as DMatrix)
   */
  def predict(testSet: DMatrix): Array[Array[Float]] = {
    _booster.predict(testSet, true, 0)
  }

  /**
   * Save the model as to HDFS-compatible file system.
   *
   * @param modelPath The model path as in Hadoop path.
   */
  def saveModelAsHadoopFile(modelPath: String): Unit = {
    val path = new Path(modelPath)
    val outputStream = path.getFileSystem(sc.hadoopConfiguration).create(path)
    _booster.saveModel(outputStream)
    outputStream.close()
  }

  /**
   * get the booster instance of this model
   */
  def booster: Booster = _booster
}
