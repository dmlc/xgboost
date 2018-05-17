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

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.scalatest.FunSuite

class XGBoostRegressorSuite extends FunSuite with PerTest {

  test("XGBoost-Spark XGBoostRegressor ouput should match XGBoost4j: regression") {
    val trainingDM = new DMatrix(Regression.train.iterator)
    val testDM = new DMatrix(Regression.test.iterator)
    val trainingDF = buildDataFrame(Regression.train)
    val testDF = buildDataFrame(Regression.test)
    val round = 5

    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "reg:linear")

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    val model2 = new XGBoostRegressor(paramMap ++ Array("num_round" -> round,
      "num_workers" -> numWorkers)).fit(trainingDF)

    val prediction2 = model2.transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[Double]("prediction"))).toMap

    assert(prediction1.indices.count { i =>
      math.abs(prediction1(i)(0) - prediction2(i)) > 0.01
    } < prediction1.length * 0.1)
  }

  test("Set params in XGBoost and MLlib way should produce same model") {
    val trainingDF = buildDataFrame(Regression.train)
    val testDF = buildDataFrame(Regression.test)
    val round = 5

    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "reg:linear",
      "num_round" -> round,
      "num_workers" -> numWorkers)

    // Set params in XGBoost way
    val model1 = new XGBoostRegressor(paramMap).fit(trainingDF)
    // Set params in MLlib way
    val model2 = new XGBoostRegressor()
      .setEta(1)
      .setMaxDepth(6)
      .setSilent(1)
      .setObjective("reg:linear")
      .setNumRound(round)
      .setNumWorkers(numWorkers)
      .fit(trainingDF)

    val prediction1 = model1.transform(testDF).select("prediction").collect()
    val prediction2 = model2.transform(testDF).select("prediction").collect()

    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
        assert(math.abs(p1 - p2) <= 0.01f)
    }
  }

  test("use weight") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:linear", "num_round" -> 5, "num_workers" -> numWorkers)

    val getWeightFromId = udf({id: Int => if (id == 0) 1.0f else 0.001f}, DataTypes.FloatType)
    val trainingDF = buildDataFrame(Regression.train)
      .withColumn("weight", getWeightFromId(col("id")))
    val testDF = buildDataFrame(Regression.test)

    val model = new XGBoostRegressor(paramMap).setWeightCol("weight").fit(trainingDF)
    val prediction = model.transform(testDF).collect()
    val first = prediction.head.getAs[Double]("prediction")
    prediction.foreach(x => assert(math.abs(x.getAs[Double]("prediction") - first) <= 0.01f))
  }
}
