/*
 Copyright (c) 2024 by Contributors

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

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class XGBoostEstimatorSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("Check for Spark encryption over-the-wire") {
    val originalSslConfOpt = ss.conf.getOption("spark.ssl.enabled")
    ss.conf.set("spark.ssl.enabled", true)

    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic")
    val training = smallBinaryClassificationVector

    withClue("xgboost-spark should throw an exception when spark.ssl.enabled = true but " +
      "xgboost.spark.ignoreSsl != true") {
      val thrown = intercept[Exception] {
        new XGBoostClassifier(paramMap).setNumRound(2).setNumWorkers(numWorkers).fit(training)
      }
      assert(thrown.getMessage.contains("xgboost.spark.ignoreSsl") &&
        thrown.getMessage.contains("spark.ssl.enabled"))
    }

    // Confirm that this check can be overridden.
    ss.conf.set("xgboost.spark.ignoreSsl", true)
    new XGBoostClassifier(paramMap).setNumRound(2).setNumWorkers(numWorkers).fit(training)

    originalSslConfOpt match {
      case None =>
        ss.conf.unset("spark.ssl.enabled")
      case Some(originalSslConf) =>
        ss.conf.set("spark.ssl.enabled", originalSslConf)
    }
    ss.conf.unset("xgboost.spark.ignoreSsl")
  }

  test("nthread configuration must be no larger than spark.task.cpus") {
    val training = smallBinaryClassificationVector
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic")
    intercept[IllegalArgumentException] {
      new XGBoostClassifier(paramMap)
        .setNumWorkers(numWorkers)
        .setNumRound(2)
        .setNthread(sc.getConf.getInt("spark.task.cpus", 1) + 1)
        .fit(training)
    }
  }

  test("test preprocess dataset") {
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(1.0, 2.0, 3.0), "a"),
      (0.0, 2, -0.5, 0.0, Vectors.dense(0.2, 1.2, 2.0), "b"),
      (2.0, 2, -0.4, -2.1, Vectors.dense(0.5, 2.2, 1.7), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setBaseMarginCol("margin")
      .setWeightCol("weight")

    val (df, indices) = classifier.preprocess(dataset)
    var schema = df.schema
    assert(!schema.names.contains("group") && !schema.names.contains("other"))
    assert(indices.labelId == schema.fieldIndex("label") &&
      indices.groupId.isEmpty &&
      indices.marginId.get == schema.fieldIndex("margin") &&
      indices.weightId.get == schema.fieldIndex("weight") &&
      indices.featureId.get == schema.fieldIndex("features") &&
      indices.featureIds.isEmpty)

    classifier.setWeightCol("")
    val (df1, indices1) = classifier.preprocess(dataset)
    schema = df1.schema
    Seq("weight", "group", "other").foreach(v => assert(!schema.names.contains(v)))
    assert(indices1.labelId == schema.fieldIndex("label") &&
      indices1.groupId.isEmpty &&
      indices1.marginId.get == schema.fieldIndex("margin") &&
      indices1.weightId.isEmpty &&
      indices1.featureId.get == schema.fieldIndex("features") &&
      indices1.featureIds.isEmpty)
  }

  test("to XGBoostLabeledPoint RDD") {
    val data = Array(
      Array(1.0, 2.0, 3.0, 4.0, 5.0),
      Array(0.0, 0.0, 0.0, 0.0, 2.0),
      Array(12.0, 13.0, 14.0, 14.0, 15.0),
      Array(20.5, 21.2, 0, 0.0, 2.0)
    )
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(data(0)), "a"),
      (2.0, 2, -0.5, 0.0, Vectors.dense(data(1)).toSparse, "b"),
      (3.0, 2, -0.5, 0.0, Vectors.dense(data(2)), "b"),
      (4.0, 2, -0.4, -2.1, Vectors.dense(data(3)), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toXGBLabeledPoint(df, indices)
    val result = rdd.collect().sortBy(x => x.label)

    assert(result.length == data.length)

    assert(result(0).label === 1.0f && result(0).baseMargin.isNaN &&
      result(0).weight === 1.0f && result(0).values === data(0).map(_.toFloat))
    assert(result(1).label == 2.0f && result(1).baseMargin.isNaN &&
      result(1).weight === 0.0f &&
      result(1).values === Vectors.dense(data(1)).toSparse.values.map(_.toFloat))
    assert(result(2).label === 3.0f && result(2).baseMargin.isNaN &&
      result(2).weight == 0.0f && result(2).values === data(2).map(_.toFloat))
    assert(result(3).label === 4.0f && result(3).baseMargin.isNaN &&
      result(3).weight === -2.1f && result(3).values === data(3).map(_.toFloat))
  }

  test("to XGBoostLabeledPoint RDD with missing Float.NaN") {
    val data = Array(
      Array(1.0, 2.0, 3.0, 4.0, 5.0),
      Array(0.0, 0.0, 0.0, 0.0, 2.0),
      Array(12.0, 13.0, Float.NaN, 14.0, 15.0),
      Array(20.5, 21.2, 0, 0.0, 2.0)
    )
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(data(0)), "a"),
      (2.0, 2, -0.5, 0.0, Vectors.dense(data(1)).toSparse, "b"),
      (3.0, 2, -0.5, 0.0, Vectors.dense(data(2)), "b"),
      (4.0, 2, -0.4, -2.1, Vectors.dense(data(3)), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toXGBLabeledPoint(df, indices)
    val result = rdd.collect().sortBy(x => x.label)

    assert(result.length == data.length - 1)

    assert(result(0).label === 1.0f && result(0).baseMargin.isNaN &&
      result(0).weight === 1.0f && result(0).values === data(0).map(_.toFloat))
    assert(result(1).label == 2.0f && result(1).baseMargin.isNaN &&
      result(1).weight === 0.0f &&
      result(1).values === Vectors.dense(data(1)).toSparse.values.map(_.toFloat))
    assert(result(2).label === 4.0f && result(2).baseMargin.isNaN &&
      result(2).weight === -2.1f && result(2).values === data(3).map(_.toFloat))
  }


  test("to XGBoostLabeledPoint RDD with missing 0.0") {
    val data = Array(
      Array(1.0, 2.0, 3.0, 4.0, 5.0),
      Array(0.0, 0.0, 0.0, 0.0, 2.0),
      Array(12.0, 13.0, Float.NaN, 14.0, 15.0),
      Array(20.5, 21.2, 0, 0.0, 2.0)
    )
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(data(0)), "a"),
      (2.0, 2, -0.5, 0.0, Vectors.dense(data(1)).toSparse, "b"),
      (3.0, 2, -0.5, 0.0, Vectors.dense(data(2)), "b"),
      (4.0, 2, -0.4, -2.1, Vectors.dense(data(3)), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")
      .setMissing(0.0f)

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toXGBLabeledPoint(df, indices)
    val result = rdd.collect().sortBy(x => x.label)

    assert(result.length == 2)

    assert(result(0).label === 1.0f && result(0).baseMargin.isNaN &&
      result(0).weight === 1.0f && result(0).values === data(0).map(_.toFloat))
    assert(result(1).label === 3.0f && result(1).baseMargin.isNaN &&
      result(1).weight == 0.0f)

    assert(result(1).values(0) === 12.0f)
    assert(result(1).values(1) === 13.0f)
    assert(result(1).values(2).isNaN)
    assert(result(1).values(3) === 14.0f)
    assert(result(1).values(4) === 15.0f)
  }

}
