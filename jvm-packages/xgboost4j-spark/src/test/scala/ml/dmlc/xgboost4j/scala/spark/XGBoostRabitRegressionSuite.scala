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

import ml.dmlc.xgboost4j.java.{Rabit, XGBoostError}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import org.apache.spark.TaskFailedListener
import org.apache.spark.SparkException
import scala.collection.JavaConverters._
import org.apache.spark.sql._
import org.scalatest.FunSuite

class XGBoostRabitRegressionSuite extends FunSuite with PerTest {
  val predictionErrorMin = 0.00001f
  val maxFailure = 2;

  override def sparkSessionBuilder: SparkSession.Builder = super.sparkSessionBuilder
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.classesToRegister", classOf[Booster].getName)
    .master(s"local[${numWorkers},${maxFailure}]")

  private def waitAndCheckSparkShutdown(waitMiliSec: Int): Boolean = {
    var totalWaitedTime = 0L
    while (!ss.sparkContext.isStopped && totalWaitedTime <= waitMiliSec) {
      Thread.sleep(10)
      totalWaitedTime += 10
    }
    return ss.sparkContext.isStopped
  }

  test("test classification prediction parity w/o ring reduce") {
    val training = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)

    val xgbSettings = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers)

    val model1 = new XGBoostClassifier(xgbSettings).fit(training)
    val prediction1 = model1.transform(testDF).select("prediction").collect()

    val model2 = new XGBoostClassifier(xgbSettings ++ Map("rabit_ring_reduce_threshold" -> 1))
      .fit(training)

    assert(Rabit.rabitEnvs.asScala.size > 3)
    Rabit.rabitEnvs.asScala.foreach( item => {
      if (item._1.toString == "rabit_reduce_ring_mincount") assert(item._2 == "1")
    })

    val prediction2 = model2.transform(testDF).select("prediction").collect()
    // check parity w/o rabit cache
    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(p1 == p2)
    }
  }

  test("test regression prediction parity w/o ring reduce") {
    val training = buildDataFrame(Regression.train)
    val testDM = new DMatrix(Regression.test.iterator, null)
    val testDF = buildDataFrame(Classification.test)
    val xgbSettings = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "reg:squarederror", "num_round" -> 5, "num_workers" -> numWorkers)
    val model1 = new XGBoostRegressor(xgbSettings).fit(training)

    val prediction1 = model1.transform(testDF).select("prediction").collect()

    val model2 = new XGBoostRegressor(xgbSettings ++ Map("rabit_ring_reduce_threshold" -> 1)
    ).fit(training)
    assert(Rabit.rabitEnvs.asScala.size > 3)
    Rabit.rabitEnvs.asScala.foreach( item => {
      if (item._1.toString == "rabit_reduce_ring_mincount") assert(item._2 == "1")
    })
    // check the equality of single instance prediction
    val prediction2 = model2.transform(testDF).select("prediction").collect()
    // check parity w/o rabit cache
    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(math.abs(p1 - p2) < predictionErrorMin)
    }
  }

  test("test rabit timeout fail handle") {
    // disable spark kill listener to verify if rabit_timeout take effect and kill tasks
    TaskFailedListener.killerStarted = true

    val training = buildDataFrame(Classification.train)
    // mock rank 0 failure during 8th allreduce synchronization
    Rabit.mockList = Array("0,8,0,0").toList.asJava

    try {
      new XGBoostClassifier(Map(
        "eta" -> "0.1",
        "max_depth" -> "10",
        "verbosity" -> "1",
        "objective" -> "binary:logistic",
        "num_round" -> 5,
        "num_workers" -> numWorkers,
        "rabit_timeout" -> 0))
        .fit(training)
    } catch {
      case e: Throwable => // swallow anything
    } finally {
      // assume all tasks throw exception almost same time
      // 100ms should be enough to exhaust all retries
      assert(waitAndCheckSparkShutdown(100) == true)
    }
  }
}
