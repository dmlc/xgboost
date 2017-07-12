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

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite

class XGBoostConfigureSuite extends FunSuite with Utils {

  test("nthread configuration must be equal to spark.task.cpus") {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite").
      set("spark.task.cpus", "4")
    val customSparkContext = new SparkContext(sparkConf)
    customSparkContext.setLogLevel("ERROR")
    // start another app
    val trainingRDD = customSparkContext.parallelize(Classification.train)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic", "nthread" -> 6)
    intercept[IllegalArgumentException] {
      XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    }
    customSparkContext.stop()
  }

  test("kryoSerializer test") {
    val eval = new EvalError()
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    val customSparkContext = new SparkContext(sparkConf)
    customSparkContext.setLogLevel("ERROR")
    val trainingRDD = customSparkContext.parallelize(Classification.train)
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(Classification.test.iterator, null))
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
    customSparkContext.stop()
  }
}
