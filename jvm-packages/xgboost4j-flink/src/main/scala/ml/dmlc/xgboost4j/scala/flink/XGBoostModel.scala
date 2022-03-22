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

package ml.dmlc.xgboost4j.scala.flink

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}

import org.apache.flink.api.scala.{DataSet, _}
import org.apache.flink.ml.math.Vector
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

class XGBoostModel (booster: Booster) extends Serializable {
  /**
    * Save the model as a Hadoop filesystem file.
    *
    * @param modelPath The model path as in Hadoop path.
    */
  def saveModelAsHadoopFile(modelPath: String): Unit = {
    booster.saveModel(FileSystem
      .get(new Configuration)
      .create(new Path(modelPath)))
  }

  /**
   * predict with the given DMatrix
   * @param testSet the local test set represented as DMatrix
   * @return prediction result
   */
  def predict(testSet: DMatrix): Array[Array[Float]] = {
    booster.predict(testSet, true, 0)
  }

  /**
    * Predict given vector dataset.
    *
    * @param data The dataset to be predicted.
    * @return The prediction result.
    */
  def predict(data: DataSet[Vector]) : DataSet[Array[Float]] = {
    val predictMap: Iterator[Vector] => Traversable[Array[Float]] =
      (it: Iterator[Vector]) => {
        val mapper = (x: Vector) => {
          val (index, value) = x.toSeq.unzip
          LabeledPoint(0.0f, x.size, index.toArray, value.map(_.toFloat).toArray)
        }
        val dataIter = for (x <- it) yield mapper(x)
        val dmat = new DMatrix(dataIter, null)
        this.booster.predict(dmat)
      }
    data.mapPartition(predictMap)
  }
}
