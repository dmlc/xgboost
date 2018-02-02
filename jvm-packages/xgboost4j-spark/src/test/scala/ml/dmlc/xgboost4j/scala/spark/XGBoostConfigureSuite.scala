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

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class XGBoostConfigureSuite extends FunSuite with PerTest {
  override def sparkSessionBuilder: SparkSession.Builder = super.sparkSessionBuilder
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryo.classesToRegister", classOf[Booster].getName)

  test("nthread configuration must be no larger than spark.task.cpus") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic",
      "nthread" -> (sc.getConf.getInt("spark.task.cpus", 1) + 1))
    intercept[IllegalArgumentException] {
      XGBoost.trainWithRDD(sc.parallelize(List()), paramMap, 5, numWorkers)
    }
  }

  test("kryoSerializer test") {
    import DataUtils._
    // TODO write an isolated test for Booster.
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator, null)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    val eval = new EvalError()
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }
}
