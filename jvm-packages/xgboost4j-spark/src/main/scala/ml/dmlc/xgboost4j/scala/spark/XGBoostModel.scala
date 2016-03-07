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

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, Booster}

class XGBoostModel(booster: Booster) extends Serializable {
  /**
    * Predict result given testRDD
    * @param testSet the testSet of Data vectors
    * @return The predicted RDD
    */
  def predict(testSet: RDD[Vector]): RDD[Array[Array[Float]]] = {
    import DataUtils._
    val broadcastBooster = testSet.sparkContext.broadcast(booster)
    val dataUtils = testSet.sparkContext.broadcast(DataUtils)
    testSet.mapPartitions { testSamples =>
      val dMatrix = new DMatrix(new JDMatrix(testSamples, null))
      Iterator(broadcastBooster.value.predict(dMatrix))
    }
  }

  def predict(testSet: DMatrix): Array[Array[Float]] = {
    booster.predict(testSet)
  }

  /**
    * Save the model as a Hadoop filesystem file.
    *
    * @param modelPath The model path as in Hadoop path.
    */
  def saveModelToHadoop(modelPath: String): Unit = {
    booster.saveModel(FileSystem
      .get(new Configuration)
      .create(new Path(modelPath)))
  }
}
