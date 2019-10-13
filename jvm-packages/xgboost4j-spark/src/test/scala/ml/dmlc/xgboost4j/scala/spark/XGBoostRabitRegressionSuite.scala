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

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}

import scala.collection.JavaConverters._
import org.apache.spark.sql._
import org.scalatest.FunSuite

class XGBoostRabitRegressionSuite extends FunSuite with PerTest {
  override def sparkSessionBuilder: SparkSession.Builder = super.sparkSessionBuilder
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.classesToRegister", classOf[Booster].getName)

  test("test parity classification prediction") {
    val training = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)

    val model1 = new XGBoostClassifier(Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers)
    ).fit(training)
    val prediction1 = model1.transform(testDF).select("prediction").collect()

    val model2 = new XGBoostClassifier(Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
      "rabit_bootstrap_cache" -> 1, "rabit_debug" -> 1, "rabit_reduce_ring_mincount" -> 100,
      "rabit_reduce_buffer" -> "2MB", "DMLC_WORKER_CONNECT_RETRY" -> 1,
      "rabit_timeout" -> 1, "rabit_timeout_sec" -> 5)).fit(training)

    assert(Rabit.rabitEnvs.asScala.size > 7)
    Rabit.rabitEnvs.asScala.foreach( item => {
      if (item._1.toString == "rabit_bootstrap_cache") assert(item._2 == "1")
      if (item._1.toString == "rabit_debug") assert(item._2 == "1")
      if (item._1.toString == "rabit_reduce_ring_mincount") assert(item._2 == "100")
      if (item._1.toString == "rabit_reduce_buffer") assert(item._2 == "2MB")
      if (item._1.toString == "dmlc_worker_connect_retry") assert(item._2 == "1")
      if (item._1.toString == "rabit_timeout") assert(item._2 == "1")
      if (item._1.toString == "rabit_timeout_sec") assert(item._2 == "5")
    })

    val prediction2 = model2.transform(testDF).select("prediction").collect()
    // check parity w/o rabit cache
    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(p1 == p2)
    }
  }

  test("test parity regression prediction") {
    val training = buildDataFrame(Regression.train)
    val testDM = new DMatrix(Regression.test.iterator, null)
    val testDF = buildDataFrame(Classification.test)

    val model1 = new XGBoostRegressor(Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "reg:squarederror", "num_round" -> 5, "num_workers" -> numWorkers)
    ).fit(training)
    val prediction1 = model1.transform(testDF).select("prediction").collect()

    val model2 = new XGBoostRegressor(Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "reg:squarederror", "num_round" -> 5, "num_workers" -> numWorkers,
      "rabit_bootstrap_cache" -> 1, "rabit_debug" -> 1, "rabit_reduce_ring_mincount" -> 100,
      "rabit_reduce_buffer" -> "2MB", "DMLC_WORKER_CONNECT_RETRY" -> 1,
      "rabit_timeout" -> 1, "rabit_timeout_sec" -> 5)).fit(training)
    assert(Rabit.rabitEnvs.asScala.size > 7)
    Rabit.rabitEnvs.asScala.foreach( item => {
      if (item._1.toString == "rabit_bootstrap_cache") assert(item._2 == "1")
      if (item._1.toString == "rabit_debug") assert(item._2 == "1")
      if (item._1.toString == "rabit_reduce_ring_mincount") assert(item._2 == "100")
      if (item._1.toString == "rabit_reduce_buffer") assert(item._2 == "2MB")
      if (item._1.toString == "dmlc_worker_connect_retry") assert(item._2 == "1")
      if (item._1.toString == "rabit_timeout") assert(item._2 == "1")
      if (item._1.toString == "rabit_timeout_sec") assert(item._2 == "5")
    })
    // check the equality of single instance prediction
    val prediction2 = model2.transform(testDF).select("prediction").collect()
    // check parity w/o rabit cache
    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(math.abs(p1 - p2) < 0.01f)
    }
  }
}
