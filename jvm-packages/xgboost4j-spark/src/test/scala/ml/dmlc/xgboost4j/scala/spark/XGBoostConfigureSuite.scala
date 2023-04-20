/*
 Copyright (c) 2014-2022 by Contributors

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

import org.apache.spark.sql._
import org.scalatest.funsuite.AnyFunSuite

class XGBoostConfigureSuite extends AnyFunSuite with PerTest {

  override def sparkSessionBuilder: SparkSession.Builder = super.sparkSessionBuilder
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryo.classesToRegister", classOf[Booster].getName)

  test("nthread configuration must be no larger than spark.task.cpus") {
    val training = buildDataFrame(Classification.train)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_workers" -> numWorkers,
      "nthread" -> (sc.getConf.getInt("spark.task.cpus", 1) + 1))
    intercept[IllegalArgumentException] {
      new XGBoostClassifier(paramMap ++ Seq("num_round" -> 2)).fit(training)
    }
  }

  test("kryoSerializer test") {
    // TODO write an isolated test for Booster.
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator, null)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers)

    val model = new XGBoostClassifier(paramMap).fit(training)
    val eval = new EvalError()
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("Check for Spark encryption over-the-wire") {
    val originalSslConfOpt = ss.conf.getOption("spark.ssl.enabled")
    ss.conf.set("spark.ssl.enabled", true)

    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 2, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)

    withClue("xgboost-spark should throw an exception when spark.ssl.enabled = true but " +
      "xgboost.spark.ignoreSsl != true") {
      val thrown = intercept[Exception] {
        new XGBoostClassifier(paramMap).fit(training)
      }
      assert(thrown.getMessage.contains("xgboost.spark.ignoreSsl") &&
        thrown.getMessage.contains("spark.ssl.enabled"))
    }

    // Confirm that this check can be overridden.
    ss.conf.set("xgboost.spark.ignoreSsl", true)
    new XGBoostClassifier(paramMap).fit(training)

    originalSslConfOpt match {
      case None =>
        ss.conf.unset("spark.ssl.enabled")
      case Some(originalSslConf) =>
        ss.conf.set("spark.ssl.enabled", originalSslConf)
    }
    ss.conf.unset("xgboost.spark.ignoreSsl")
  }
}
