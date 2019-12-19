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
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite

import scala.util.Random

class MissingValueHandlingSuite extends FunSuite with PerTest {
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
    org.apache.spark.SPARK_VERSION match {
      case version if version.startsWith("2.4") =>
        val m = vectorAssembler.getClass.getDeclaredMethods
          .filter(_.getName.contains("setHandleInvalid")).head
        m.invoke(vectorAssembler, "keep")
      case _ =>
    }
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
    intercept[XGBoostError] {
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
    intercept[XGBoostError] {
      new XGBoostClassifier(paramMap).fit(inputDF)
    }
  }

  test("specify a non-zero missing value but set allow_non_zero_missing " +
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
}
