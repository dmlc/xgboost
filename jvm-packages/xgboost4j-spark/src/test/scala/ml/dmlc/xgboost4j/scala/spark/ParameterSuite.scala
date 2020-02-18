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

import ml.dmlc.xgboost4j.java.XGBoostError
import org.scalatest.{BeforeAndAfterAll, FunSuite, Ignore}

import org.apache.spark.ml.param.ParamMap

class ParameterSuite extends FunSuite with PerTest with BeforeAndAfterAll {

  test("XGBoost and Spark parameters synchronize correctly") {
    val xgbParamMap = Map("eta" -> "1", "objective" -> "binary:logistic",
      "objective_type" -> "classification")
    // from xgboost params to spark params
    val xgb = new XGBoostClassifier(xgbParamMap)
    assert(xgb.getEta === 1.0)
    assert(xgb.getObjective === "binary:logistic")
    assert(xgb.getObjectiveType === "classification")
    // from spark to xgboost params
    val xgbCopy = xgb.copy(ParamMap.empty)
    assert(xgbCopy.MLlib2XGBoostParams("eta").toString.toDouble === 1.0)
    assert(xgbCopy.MLlib2XGBoostParams("objective").toString === "binary:logistic")
    assert(xgbCopy.MLlib2XGBoostParams("objective_type").toString === "classification")
    val xgbCopy2 = xgb.copy(ParamMap.empty.put(xgb.evalMetric, "logloss"))
    assert(xgbCopy2.MLlib2XGBoostParams("eval_metric").toString === "logloss")
  }

  private def waitForSparkContextShutdown(): Unit = {
    var totalWaitedTime = 0L
    while (!ss.sparkContext.isStopped && totalWaitedTime <= 120000) {
      Thread.sleep(10000)
      totalWaitedTime += 10000
    }
    assert(ss.sparkContext.isStopped === true)
  }

  test("fail training elegantly with unsupported objective function") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "wrong_objective_function", "num_class" -> "6", "num_round" -> 5,
      "num_workers" -> numWorkers)
    val trainingDF = buildDataFrame(MultiClassification.train)
    val xgb = new XGBoostClassifier(paramMap)
    try {
      val model = xgb.fit(trainingDF)
    } catch {
      case e: Throwable => // swallow anything
    } finally {
      waitForSparkContextShutdown()
    }
  }

  test("fail training elegantly with unsupported eval metrics") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6", "num_round" -> 5,
      "num_workers" -> numWorkers, "eval_metric" -> "wrong_eval_metrics")
    val trainingDF = buildDataFrame(MultiClassification.train)
    val xgb = new XGBoostClassifier(paramMap)
    try {
      val model = xgb.fit(trainingDF)
    } catch {
      case e: Throwable => // swallow anything
    } finally {
      waitForSparkContextShutdown()
    }
  }
}
