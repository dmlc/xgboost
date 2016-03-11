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
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, Booster}

class XGBoostModel(booster: Booster)(implicit val sc: SparkContext) extends Serializable {

  /**
   * Predict result with the given testset (represented as RDD)
   */
  def predict(testSet: RDD[Vector]): RDD[Array[Array[Float]]] = {
    import DataUtils._
    val broadcastBooster = testSet.sparkContext.broadcast(booster)
    testSet.mapPartitions { testSamples =>
      val dMatrix = new DMatrix(new JDMatrix(testSamples, null))
      Iterator(broadcastBooster.value.predict(dMatrix))
    }
  }

  /**
   * predict result given the test data (represented as DMatrix)
   */
  def predict(testSet: DMatrix): Array[Array[Float]] = {
    booster.predict(testSet, true, 0)
  }

  /**
   * Save the model as to HDFS-compatible file system.
   *
   * @param modelPath The model path as in Hadoop path.
   */
  def saveModelAsHadoopFile(modelPath: String): Unit = {
    val path = new Path(modelPath)
    val outputStream = path.getFileSystem(sc.hadoopConfiguration).create(path)
    booster.saveModel(outputStream)
    outputStream.close()
  }
}
