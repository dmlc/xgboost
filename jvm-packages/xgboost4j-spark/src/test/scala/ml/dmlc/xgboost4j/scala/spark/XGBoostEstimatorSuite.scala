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

import java.io.File
import java.util.Arrays

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.DMatrix

class XGBoostEstimatorSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("RuntimeParameter") {
    var runtimeParams = new XGBoostClassifier(
      Map("device" -> "cpu", "num_workers" -> 1, "num_round" -> 1))
      .getRuntimeParameters(true)
    assert(!runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cuda", "num_workers" -> 1, "num_round" -> 1))
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cpu", "tree_method" -> "gpu_hist", "num_workers" -> 1, "num_round" -> 1))
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cuda", "tree_method" -> "gpu_hist",
        "num_workers" -> 1, "num_round" -> 1))
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)
  }

  test("custom_eval does not support early stopping") {
    val paramMap = Map("eta" -> "0.1", "custom_eval" -> new EvalError, "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6", "num_round" -> 5,
      "num_workers" -> numWorkers, "num_early_stopping_rounds" -> 2)

    val trainingDF = smallBinaryClassificationVector

    val thrown = intercept[IllegalArgumentException] {
      new XGBoostClassifier(paramMap).fit(trainingDF)
    }

    assert(thrown.getMessage.contains("custom_eval does not support early stopping"))
  }

  test("test persistence of XGBoostClassifier and XGBoostClassificationModel " +
    "using custom Eval and Obj") {
    val trainingDF = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)

    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6",
      "verbosity" -> "1", "objective" -> "binary:logistic")

    val xgbc = new XGBoostClassifier(paramMap)
      .setCustomObj(new CustomObj(1))
      .setCustomEval(new EvalError)
      .setNumRound(10)
      .setNumWorkers(numWorkers)

    val xgbcPath = new File(tempDir.toFile, "xgbc").getPath
    xgbc.write.overwrite().save(xgbcPath)
    val xgbc2 = XGBoostClassifier.load(xgbcPath)

    assert(xgbc.getCustomObj.asInstanceOf[CustomObj].customParameter === 1)
    assert(xgbc2.getCustomObj.asInstanceOf[CustomObj].customParameter === 1)

    val eval = new EvalError()

    val model = xgbc.fit(trainingDF)
    val evalResults = eval.eval(model.nativeBooster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults < 0.1)
    val xgbcModelPath = new File(tempDir.toFile, "xgbcModel").getPath
    model.write.overwrite.save(xgbcModelPath)
    val model2 = XGBoostClassificationModel.load(xgbcModelPath)
    assert(Arrays.equals(model.nativeBooster.toByteArray, model2.nativeBooster.toByteArray))

    assert(model.getEta === model2.getEta)
    assert(model.getNumRound === model2.getNumRound)
    assert(model.getRawPredictionCol === model2.getRawPredictionCol)
    val evalResults2 = eval.eval(model2.nativeBooster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults === evalResults2)
  }

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
      .setNumWorkers(2)

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
      .setNumWorkers(2)

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
      .setBaseMarginCol("margin")
      .setNumWorkers(2)
      .setMissing(0.0f)

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toXGBLabeledPoint(df, indices)
    val result = rdd.collect().sortBy(x => x.label)

    assert(result.length == 2)

    assert(result(0).label === 1.0f && result(0).baseMargin === 0.5f &&
      result(0).weight === 1.0f && result(0).values === data(0).map(_.toFloat))
    assert(result(1).label === 3.0f && result(1).baseMargin === -0.5f &&
      result(1).weight == 0.0f)

    assert(result(1).values(0) === 12.0f)
    assert(result(1).values(1) === 13.0f)
    assert(result(1).values(2).isNaN)
    assert(result(1).values(3) === 14.0f)
    assert(result(1).values(4) === 15.0f)
  }

  test("test to RDD watches") {
    val data = Array(
      Array(1.0, 2.0, 3.0, 4.0, 5.0),
      Array(0.0, 0.0, 0.0, 0.0, 2.0),
      Array(12.0, 13.0, 14.0, 14.0, 15.0),
      Array(20.5, 21.2, 0.0, 0.0, 2.0)
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
      .setBaseMarginCol("margin")
      .setNumWorkers(2)

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toRdd(df, indices)
    val result = rdd.mapPartitions { iter =>
      if (iter.hasNext) {
        val watches = iter.next()
        val size = watches.size
        val rowNum = watches.datasets(0).rowNum
        val labels = watches.datasets(0).getLabel
        val weight = watches.datasets(0).getWeight
        val margins = watches.datasets(0).getBaseMargin
        watches.delete()
        Iterator.single((size, rowNum, labels, weight, margins))
      } else {
        Iterator.empty
      }
    }.collect()

    val labels: ArrayBuffer[Float] = ArrayBuffer.empty
    val weight: ArrayBuffer[Float] = ArrayBuffer.empty
    val margins: ArrayBuffer[Float] = ArrayBuffer.empty

    var totalRows = 0L
    for (row <- result) {
      assert(row._1 === 1)
      totalRows = totalRows + row._2
      labels.append(row._3: _*)
      weight.append(row._4: _*)
      margins.append(row._5: _*)
    }
    assert(totalRows === 4)
    assert(labels.toArray.sorted === Array(1.0f, 2.0f, 3.0f, 4.0f).sorted)
    assert(weight.toArray.sorted === Array(0.0f, 0.0f, 1.0f, -2.1f).sorted)
    assert(margins.toArray.sorted === Array(-0.5f, -0.5f, -0.4f, 0.5f).sorted)

  }

  test("test to RDD watches with eval") {
    val trainData = Array(
      Array(-1.0, -2.0, -3.0, -4.0, -5.0),
      Array(2.0, 2.0, 2.0, 3.0, -2.0),
      Array(-12.0, -13.0, -14.0, -14.0, -15.0),
      Array(-20.5, -21.2, 0.0, 0.0, 2.0)
    )
    val trainDataset = ss.createDataFrame(sc.parallelize(Seq(
      (11.0, 0, 0.15, 11.0, Vectors.dense(trainData(0)), "a"),
      (12.0, 12, -0.15, 10.0, Vectors.dense(trainData(1)).toSparse, "b"),
      (13.0, 12, -0.15, 10.0, Vectors.dense(trainData(2)), "b"),
      (14.0, 12, -0.14, -12.1, Vectors.dense(trainData(3)), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")
    val evalData = Array(
      Array(1.0, 2.0, 3.0, 4.0, 5.0),
      Array(0.0, 0.0, 0.0, 0.0, 2.0),
      Array(12.0, 13.0, 14.0, 14.0, 15.0),
      Array(20.5, 21.2, 0.0, 0.0, 2.0)
    )
    val evalDataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(evalData(0)), "a"),
      (2.0, 2, -0.5, 0.0, Vectors.dense(evalData(1)).toSparse, "b"),
      (3.0, 2, -0.5, 0.0, Vectors.dense(evalData(2)), "b"),
      (4.0, 2, -0.4, -2.1, Vectors.dense(evalData(3)), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")
      .setBaseMarginCol("margin")
      .setEvalDataset(evalDataset)
      .setNumWorkers(2)

    val (df, indices) = classifier.preprocess(trainDataset)
    val rdd = classifier.toRdd(df, indices)
    val result = rdd.mapPartitions { iter =>
      if (iter.hasNext) {
        val watches = iter.next()
        val size = watches.size
        val rowNum = watches.datasets(1).rowNum
        val labels = watches.datasets(1).getLabel
        val weight = watches.datasets(1).getWeight
        val margins = watches.datasets(1).getBaseMargin
        watches.delete()
        Iterator.single((size, rowNum, labels, weight, margins))
      } else {
        Iterator.empty
      }
    }.collect()

    val labels: ArrayBuffer[Float] = ArrayBuffer.empty
    val weight: ArrayBuffer[Float] = ArrayBuffer.empty
    val margins: ArrayBuffer[Float] = ArrayBuffer.empty

    var totalRows = 0L
    for (row <- result) {
      assert(row._1 === 2)
      totalRows = totalRows + row._2
      labels.append(row._3: _*)
      weight.append(row._4: _*)
      margins.append(row._5: _*)
    }
    assert(totalRows === 4)
    assert(labels.toArray.sorted === Array(1.0f, 2.0f, 3.0f, 4.0f).sorted)
    assert(weight.toArray.sorted === Array(0.0f, 0.0f, 1.0f, -2.1f).sorted)
    assert(margins.toArray.sorted === Array(-0.5f, -0.5f, -0.4f, 0.5f).sorted)
  }

}
