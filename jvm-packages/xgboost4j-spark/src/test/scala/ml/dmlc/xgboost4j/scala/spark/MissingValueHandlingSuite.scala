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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

import org.apache.spark.SparkException

class MissingValueHandlingSuite extends AnyFunSuite with PerTest {
  test("dense vectors containing missing value") {
    def buildDenseDataFrame(): DataFrame = {
      val numRows = 100
      val numCols = 5
      val data = (0 until numRows).map { x =>
        val label = Random.nextInt(2)
        val values = Array.tabulate[Double](numCols) { c =>
          if (c == numCols - 1) 0 else Random.nextDouble
        }
        (label, Vectors.dense(values))
      }
      ss.createDataFrame(sc.parallelize(data.toList)).toDF("label", "features")
    }
    val denseDF = buildDenseDataFrame().repartition(4)
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> 0, "num_workers" -> numWorkers).toMap
    val model = new XGBoostClassifier(paramMap).fit(denseDF)
    model.transform(denseDF).collect()
  }

  test("handle Float.NaN as missing value correctly") {
    val spark = ss
    import spark.implicits._
    val testDF = Seq(
      (1.0f, 0.0f, Float.NaN, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (1.0f, Float.NaN, 0.0f, 0.0),
      (0.0f, 1.0f, 0.0f, 1.0),
      (Float.NaN, 0.0f, 0.0f, 1.0)
    ).toDF("col1", "col2", "col3", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> Float.NaN, "num_workers" -> 1).toMap
    val model = new XGBoostClassifier(paramMap).fit(inputDF)
    model.transform(inputDF).collect()
  }

  test("specify a non-zero missing value but with dense vector does not stop" +
    " application") {
    val spark = ss
    import spark.implicits._
    // spark uses 1.5 * (nnz + 1.0) < size as the condition to decide whether using sparse or dense
    // vector,
    val testDF = Seq(
      (1.0f, 0.0f, -1.0f, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (1.0f, -1.0f, 0.0f, 0.0),
      (0.0f, 1.0f, 0.0f, 1.0),
      (-1.0f, 0.0f, 0.0f, 1.0)
    ).toDF("col1", "col2", "col3", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol("features")
    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> -1.0f, "num_workers" -> 1).toMap
    val model = new XGBoostClassifier(paramMap).fit(inputDF)
    model.transform(inputDF).collect()
  }

  test("specify a non-zero missing value and meet an empty vector we should" +
    " stop the application") {
    val spark = ss
    import spark.implicits._
    val testDF = Seq(
      (1.0f, 0.0f, -1.0f, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0),
      (1.0f, -1.0f, 0.0f, 0.0),
      (0.0f, 0.0f, 0.0f, 1.0),// empty vector
      (-1.0f, 0.0f, 0.0f, 1.0)
    ).toDF("col1", "col2", "col3", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol("features")
    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> -1.0f, "num_workers" -> 1).toMap
    intercept[SparkException] {
      new XGBoostClassifier(paramMap).fit(inputDF)
    }
  }

  test("specify a non-zero missing value and meet a Sparse vector we should" +
    " stop the application") {
    val spark = ss
    import spark.implicits._
    // spark uses 1.5 * (nnz + 1.0) < size as the condition to decide whether using sparse or dense
    // vector,
    val testDF = Seq(
      (1.0f, 0.0f, -1.0f, 1.0f, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 1.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (1.0f, -1.0f, 0.0f, 1.0f, 0.0),
      (0.0f, 0.0f, 0.0f, 1.0f, 1.0),
      (-1.0f, 0.0f, 0.0f, 1.0f, 1.0)
    ).toDF("col1", "col2", "col3", "col4", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3", "col4"))
      .setOutputCol("features")
    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    inputDF.show()
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> -1.0f, "num_workers" -> 1).toMap
    intercept[SparkException] {
      new XGBoostClassifier(paramMap).fit(inputDF)
    }
  }

  test("specify a non-zero missing value but set allow_non_zero_for_missing " +
    "does not stop application") {
    val spark = ss
    import spark.implicits._
    // spark uses 1.5 * (nnz + 1.0) < size as the condition to decide whether using sparse or dense
    // vector,
    val testDF = Seq(
      (7.0f, 0.0f, -1.0f, 1.0f, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 1.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (1.0f, -1.0f, 0.0f, 1.0f, 0.0),
      (0.0f, 0.0f, 0.0f, 1.0f, 1.0),
      (-1.0f, 0.0f, 0.0f, 1.0f, 1.0)
    ).toDF("col1", "col2", "col3", "col4", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3", "col4"))
      .setOutputCol("features")
    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    inputDF.show()
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> -1.0f,
      "num_workers" -> 1, "allow_non_zero_for_missing" -> "true").toMap
    val model = new XGBoostClassifier(paramMap).fit(inputDF)
    model.transform(inputDF).collect()
  }

  // https://github.com/dmlc/xgboost/pull/5929
  test("handle the empty last row correctly with a missing value as 0") {
    val spark = ss
    import spark.implicits._
    // spark uses 1.5 * (nnz + 1.0) < size as the condition to decide whether using sparse or dense
    // vector,
    val testDF = Seq(
      (7.0f, 0.0f, -1.0f, 1.0f, 1.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (0.0f, 1.0f, 0.0f, 1.0f, 0.0),
      (1.0f, 0.0f, 1.0f, 1.0f, 1.0),
      (1.0f, -1.0f, 0.0f, 1.0f, 0.0),
      (0.0f, 0.0f, 0.0f, 1.0f, 1.0),
      (0.0f, 0.0f, 0.0f, 0.0f, 0.0)
    ).toDF("col1", "col2", "col3", "col4", "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3", "col4"))
      .setOutputCol("features")
    val inputDF = vectorAssembler.transform(testDF).select("features", "label")
    inputDF.show()
    val paramMap = List("eta" -> "1", "max_depth" -> "2",
      "objective" -> "binary:logistic", "missing" -> 0.0f,
      "num_workers" -> 1, "allow_non_zero_for_missing" -> "true").toMap
    val model = new XGBoostClassifier(paramMap).fit(inputDF)
    model.transform(inputDF).collect()
  }

  test("Getter and setter for AllowNonZeroForMissingValue works") {
    {
      val paramMap = Map("eta" -> "1", "max_depth" -> "6",
        "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers)
      val training = buildDataFrame(Classification.train)
      val classifier = new XGBoostClassifier(paramMap)
      classifier.setAllowNonZeroForMissing(true)
      assert(classifier.getAllowNonZeroForMissingValue)
      classifier.setAllowNonZeroForMissing(false)
      assert(!classifier.getAllowNonZeroForMissingValue)
      val model = classifier.fit(training)
      model.setAllowNonZeroForMissing(true)
      assert(model.getAllowNonZeroForMissingValue)
      model.setAllowNonZeroForMissing(false)
      assert(!model.getAllowNonZeroForMissingValue)
    }

    {
      val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
        "objective" -> "reg:squarederror", "num_round" -> 5, "num_workers" -> numWorkers)
      val training = buildDataFrame(Regression.train)
      val regressor = new XGBoostRegressor(paramMap)
      regressor.setAllowNonZeroForMissing(true)
      assert(regressor.getAllowNonZeroForMissingValue)
      regressor.setAllowNonZeroForMissing(false)
      assert(!regressor.getAllowNonZeroForMissingValue)
      val model = regressor.fit(training)
      model.setAllowNonZeroForMissing(true)
      assert(model.getAllowNonZeroForMissingValue)
      model.setAllowNonZeroForMissing(false)
      assert(!model.getAllowNonZeroForMissingValue)
    }
  }
}
