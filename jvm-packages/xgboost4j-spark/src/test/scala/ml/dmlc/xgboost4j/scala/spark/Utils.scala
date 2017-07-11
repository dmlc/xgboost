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

import java.io.File

import scala.collection.mutable.ListBuffer
import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.rdd.RDD

trait Utils extends Serializable {
  protected val numWorkers: Int = Runtime.getRuntime.availableProcessors()

  protected def cleanExternalCache(prefix: String): Unit = {
    val dir = new File(".")
    for (file <- dir.listFiles() if file.getName.startsWith(prefix)) {
      file.delete()
    }
  }

  protected def buildTrainingRDD(sparkContext: SparkContext): RDD[LabeledPoint] = {
    sparkContext.parallelize(Agaricus.train, numWorkers)
  }
}
