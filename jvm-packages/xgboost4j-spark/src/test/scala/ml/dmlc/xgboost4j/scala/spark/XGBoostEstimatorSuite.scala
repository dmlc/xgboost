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

import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DoubleType, FloatType}
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.parseJson
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.Utils.TRAIN_NAME

class XGBoostEstimatorSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("params") {
    val df = smallBinaryClassificationVector
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "objective" -> "binary:logistic"
    )
    val estimator = new XGBoostClassifier(xgbParams)
      .setFeaturesCol("features")
      .setMissing(0.2f)
      .setAlpha(0.97)
      .setLeafPredictionCol("leaf")
      .setContribPredictionCol("contrib")
      .setNumRound(1)

    assert(estimator.getMaxDepth === 5)
    assert(estimator.getEta === 0.2)
    assert(estimator.getObjective === "binary:logistic")
    assert(estimator.getFeaturesCol === "features")
    assert(estimator.getMissing === 0.2f)
    assert(estimator.getAlpha === 0.97)

    estimator.setEta(0.66).setMaxDepth(7)
    assert(estimator.getMaxDepth === 7)
    assert(estimator.getEta === 0.66)

    val model = estimator.fit(df)
    assert(model.getMaxDepth === 7)
    assert(model.getEta === 0.66)
    assert(model.getObjective === "binary:logistic")
    assert(model.getFeaturesCol === "features")
    assert(model.getMissing === 0.2f)
    assert(model.getAlpha === 0.97)
    assert(model.getLeafPredictionCol === "leaf")
    assert(model.getContribPredictionCol === "contrib")
  }

  test("camel case parameters") {
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "featuresCol" -> "abc",
      "num_workers" -> 2,
      "numRound" -> 11
    )
    val estimator = new XGBoostClassifier(xgbParams)
    assert(estimator.getFeaturesCol === "abc")
    assert(estimator.getNumWorkers === 2)
    assert(estimator.getNumRound === 11)
    assert(estimator.getMaxDepth === 5)

    val xgbParams1: Map[String, Any] = Map(
      "maxDepth" -> 5,
      "features_col" -> "abc",
      "numWorkers" -> 2,
      "num_round" -> 11
    )
    val estimator1 = new XGBoostClassifier(xgbParams1)
    assert(estimator1.getFeaturesCol === "abc")
    assert(estimator1.getNumWorkers === 2)
    assert(estimator1.getNumRound === 11)
    assert(estimator1.getMaxDepth === 5)
  }

  test("get xgboost parameters") {
    val params: Map[String, Any] = Map(
      "max_depth" -> 5,
      "featuresCol" -> "abc",
      "label" -> "class",
      "num_workers" -> 2,
      "tree_method" -> "hist",
      "numRound" -> 11,
      "not_exist_parameters" -> "hello"
    )
    val estimator = new XGBoostClassifier(params)
    val xgbParams = estimator.getXGBoostParams
    assert(xgbParams.size === 2)
    assert(xgbParams.contains("max_depth") && xgbParams.contains("tree_method"))
  }

  test("nthread") {
    val classifier = new XGBoostClassifier().setNthread(100)

    intercept[IllegalArgumentException](
      classifier.validate(smallBinaryClassificationVector)
    )
  }

  test("RuntimeParameter") {
    var runtimeParams = new XGBoostClassifier(
      Map("device" -> "cpu"))
      .getRuntimeParameters(true)
    assert(!runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cuda")).setNumWorkers(1).setNumRound(1)
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cpu", "tree_method" -> "gpu_hist")).setNumWorkers(1).setNumRound(1)
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)

    runtimeParams = new XGBoostClassifier(
      Map("device" -> "cuda", "tree_method" -> "gpu_hist")).setNumWorkers(1).setNumRound(1)
      .getRuntimeParameters(true)
    assert(runtimeParams.runOnGpu)
  }

  test("missing value exception for sparse vector") {
    val sparse1 = Vectors.dense(0.0, 0.0, 0.0).toSparse
    assert(sparse1.isInstanceOf[SparseVector])
    val sparse2 = Vectors.dense(0.5, 2.2, 1.7).toSparse
    assert(sparse2.isInstanceOf[SparseVector])

    val sparseInput = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, sparse1),
      (2.0, sparse2)
    ))).toDF("label", "features")

    val classifier = new XGBoostClassifier()
    val (input, columnIndexes) = classifier.preprocess(sparseInput)
    val rdd = classifier.toXGBLabeledPoint(input, columnIndexes)

    val exception = intercept[SparkException] {
      rdd.collect()
    }
    assert(exception.getMessage.contains("We've detected sparse vectors in the dataset " +
      "that need conversion to dense format"))

    // explicitly set missing value, no exception
    classifier.setMissing(Float.NaN)
    val rdd1 = classifier.toXGBLabeledPoint(input, columnIndexes)
    rdd1.collect()
  }

  test("missing value for dense vector no need to set missing explicitly") {
    val dense1 = Vectors.dense(0.0, 0.0, 0.0)
    assert(dense1.isInstanceOf[DenseVector])
    val dense2 = Vectors.dense(0.5, 2.2, 1.7)
    assert(dense2.isInstanceOf[DenseVector])

    val sparseInput = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, dense1),
      (2.0, dense2)
    ))).toDF("label", "features")

    val classifier = new XGBoostClassifier()
    val (input, columnIndexes) = classifier.preprocess(sparseInput)
    val rdd = classifier.toXGBLabeledPoint(input, columnIndexes)
    rdd.collect()
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

  test("preprocess dataset") {
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(1.0, 2.0, 3.0), "a"),
      (0.0, 2, -0.5, 0.0, Vectors.dense(0.2, 1.2, 2.0), "b"),
      (2.0, 2, -0.4, -2.1, Vectors.dense(0.5, 2.2, 1.7), "c")
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
      Array(20.5, 21.2, 0.0, 0.0, 2.0)
    )
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(data(0)), "a"),
      (2.0, 2, -0.5, 0.0, Vectors.dense(data(1)).toSparse, "b"),
      (3.0, 2, -0.5, 0.0, Vectors.dense(data(2)), "b"),
      (4.0, 2, -0.4, -2.1, Vectors.dense(data(3)), "c")
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")
      .setNumWorkers(2)
      .setMissing(Float.NaN)

    val (df, indices) = classifier.preprocess(dataset)
    val rdd = classifier.toXGBLabeledPoint(df, indices)
    val result = rdd.collect().sortBy(x => x.label)

    assert(result.length == data.length)

    def toArray(index: Int): Array[Float] = {
      val labelPoint = result(index)
      if (labelPoint.indices != null) {
        Vectors.sparse(labelPoint.size,
          labelPoint.indices,
          labelPoint.values.map(_.toDouble)).toArray.map(_.toFloat)
      } else {
        labelPoint.values
      }
    }

    assert(result(0).label === 1.0f && result(0).baseMargin.isNaN &&
      result(0).weight === 1.0f && toArray(0) === data(0).map(_.toFloat))
    assert(result(1).label == 2.0f && result(1).baseMargin.isNaN &&
      result(1).weight === 0.0f && toArray(1) === data(1).map(_.toFloat))
    assert(result(2).label === 3.0f && result(2).baseMargin.isNaN &&
      result(2).weight == 0.0f && toArray(2) === data(2).map(_.toFloat))
    assert(result(3).label === 4.0f && result(3).baseMargin.isNaN &&
      result(3).weight === -2.1f && toArray(3) === data(3).map(_.toFloat))
  }

  Seq((Float.NaN, 2), (0.0f, 7 + 2), (15.0f, 1 + 2), (10101011.0f, 0 + 2)).foreach {
    case (missing, expectedMissingValue) =>
      test(s"to RDD watches with missing $missing") {
        val data = Array(
          Array(1.0, 2.0, 3.0, 4.0, 5.0),
          Array(1.0, Float.NaN, 0.0, 0.0, 2.0),
          Array(12.0, 13.0, Float.NaN, 14.0, 15.0),
          Array(0.0, 0.0, 0.0, 0.0, 0.0)
        )
        val dataset = ss.createDataFrame(sc.parallelize(Seq(
          (1.0, 0, 0.5, 1.0, Vectors.dense(data(0)), "a"),
          (2.0, 2, -0.5, 0.0, Vectors.dense(data(1)).toSparse, "b"),
          (3.0, 3, -0.5, 0.0, Vectors.dense(data(2)), "b"),
          (4.0, 4, -0.4, -2.1, Vectors.dense(data(3)), "c")
        ))).toDF("label", "group", "margin", "weight", "features", "other")

        val classifier = new XGBoostClassifier()
          .setLabelCol("label")
          .setFeaturesCol("features")
          .setWeightCol("weight")
          .setBaseMarginCol("margin")
          .setMissing(missing)
          .setNumWorkers(2)

        val (df, indices) = classifier.preprocess(dataset)
        val rdd = classifier.toRdd(df, indices)
        val result = rdd.mapPartitions { iter =>
          if (iter.hasNext) {
            val watches = iter.next()
            val size = watches.size
            val trainDM = watches.toMap(TRAIN_NAME)
            val rowNum = trainDM.rowNum
            val labels = trainDM.getLabel
            val weight = trainDM.getWeight
            val margins = trainDM.getBaseMargin
            val nonMissing = trainDM.nonMissingNum
            watches.delete()
            Iterator.single((size, rowNum, labels, weight, margins, nonMissing))
          } else {
            Iterator.empty
          }
        }.collect()

        val labels: ArrayBuffer[Float] = ArrayBuffer.empty
        val weight: ArrayBuffer[Float] = ArrayBuffer.empty
        val margins: ArrayBuffer[Float] = ArrayBuffer.empty
        var nonMissingValues = 0L
        var totalRows = 0L

        for (row <- result) {
          assert(row._1 === 1)
          totalRows = totalRows + row._2
          labels.append(row._3: _*)
          weight.append(row._4: _*)
          margins.append(row._5: _*)
          nonMissingValues = nonMissingValues + row._6
        }
        assert(totalRows === 4)
        assert(nonMissingValues === data.size * data(0).length - expectedMissingValue)
        assert(labels.toArray.sorted === Array(1.0f, 2.0f, 3.0f, 4.0f).sorted)
        assert(weight.toArray.sorted === Array(0.0f, 0.0f, 1.0f, -2.1f).sorted)
        assert(margins.toArray.sorted === Array(-0.5f, -0.5f, -0.4f, 0.5f).sorted)
      }
  }

  test("to RDD watches with eval") {
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
      (14.0, 12, -0.14, -12.1, Vectors.dense(trainData(3)), "c")
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
      (4.0, 2, -0.4, -2.1, Vectors.dense(evalData(3)), "c")
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("weight")
      .setBaseMarginCol("margin")
      .setEvalDataset(evalDataset)
      .setNumWorkers(2)
      .setMissing(Float.NaN)

    val (df, indices) = classifier.preprocess(trainDataset)
    val rdd = classifier.toRdd(df, indices)
    val result = rdd.mapPartitions { iter =>
      if (iter.hasNext) {
        val watches = iter.next()
        val size = watches.size
        val evalDM = watches.toMap(Utils.VALIDATION_NAME)
        val rowNum = evalDM.rowNum
        val labels = evalDM.getLabel
        val weight = evalDM.getWeight
        val margins = evalDM.getBaseMargin
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

  test("XGBoost-Spark model format should match xgboost4j") {
    val trainingDF = buildDataFrame(MultiClassification.train)

    Seq(new XGBoostClassifier()).foreach { est =>
      est.setNumRound(5)
      val model = est.fit(trainingDF)

      // test json
      val modelPath = new File(tempDir.toFile, "xgbc").getPath
      model.write.overwrite().option("format", "json").save(modelPath)
      val nativeJsonModelPath = new File(tempDir.toFile, "nativeModel.json").getPath
      model.nativeBooster.saveModel(nativeJsonModelPath)
      assert(compareTwoFiles(new File(modelPath, "data/model").getPath,
        nativeJsonModelPath))

      // test ubj
      val modelUbjPath = new File(tempDir.toFile, "xgbcUbj").getPath
      model.write.overwrite().save(modelUbjPath)
      val nativeUbjModelPath = new File(tempDir.toFile, "nativeModel.ubj").getPath
      model.nativeBooster.saveModel(nativeUbjModelPath)
      assert(compareTwoFiles(new File(modelUbjPath, "data/model").getPath,
        nativeUbjModelPath))

      // json file should be indifferent with ubj file
      val modelJsonPath = new File(tempDir.toFile, "xgbcJson").getPath
      model.write.overwrite().option("format", "json").save(modelJsonPath)
      val nativeUbjModelPath1 = new File(tempDir.toFile, "nativeModel1.ubj").getPath
      model.nativeBooster.saveModel(nativeUbjModelPath1)
      assert(!compareTwoFiles(new File(modelJsonPath, "data/model").getPath,
        nativeUbjModelPath1))
    }
  }

  test("native json model file should store feature_name and feature_type") {
    val featureNames = (1 to 33).map(idx => s"feature_${idx}").toArray
    val featureTypes = (1 to 33).map(idx => "q").toArray
    val trainingDF = buildDataFrame(MultiClassification.train)
    val xgb = new XGBoostClassifier()
      .setNumWorkers(numWorkers)
      .setFeatureNames(featureNames)
      .setFeatureTypes(featureTypes)
      .setNumRound(2)
    val model = xgb.fit(trainingDF)
    val modelStr = new String(model.nativeBooster.toByteArray("json"))
    val jsonModel = parseJson(modelStr)
    implicit val formats: Formats = DefaultFormats
    val featureNamesInModel = (jsonModel \ "learner" \ "feature_names").extract[List[String]]
    val featureTypesInModel = (jsonModel \ "learner" \ "feature_types").extract[List[String]]
    assert(featureNamesInModel.length == 33)
    assert(featureTypesInModel.length == 33)
    assert(featureNames sameElements featureNamesInModel)
    assert(featureTypes sameElements featureTypesInModel)
  }

  test("Exception with clear message") {
    val df = smallMultiClassificationVector
    val classifier = new XGBoostClassifier()
      .setNumRound(2)
      .setObjective("multi:softprob")
      .setNumClass(2)

    val exception = intercept[SparkException] {
      classifier.fit(df)
    }

    exception.getMessage.contains("SoftmaxMultiClassObj: label must be in [0, num_class).")
  }

  test("Support array(float)") {
    val df = smallBinaryClassificationArray
    val matched = df.schema("features").dataType match {
      case ArrayType(DoubleType, _) => true
      case _ => false
    }
    assert(matched)

    val newDf = df.withColumn("features", col("features").cast(ArrayType(FloatType)))
    val matched1 = newDf.schema("features").dataType match {
      case ArrayType(FloatType, _) => true
      case _ => false
    }
    assert(matched1)

    val classifier = new XGBoostClassifier()
    assert(classifier.featureIsArrayType(df.schema))

    val (processed, _) = classifier.preprocess(df)
    val matched2 = processed.schema("features").dataType match {
      case ArrayType(FloatType, _) => true
      case _ => false
    }
    assert(matched2)
  }

  test("Support array(double)") {
    val df = smallBinaryClassificationArray
    val matched = df.schema("features").dataType match {
      case ArrayType(DoubleType, _) => true
      case _ => false
    }
    assert(matched)

    val classifier = new XGBoostClassifier()
    assert(classifier.featureIsArrayType(df.schema))

    val (processed, _) = classifier.preprocess(df)
    val matched1 = processed.schema("features").dataType match {
      case ArrayType(FloatType, _) => true
      case _ => false
    }
    assert(matched1)
  }

  test("Fit and transform with array type") {
    val df = smallBinaryClassificationArray
    val classifier = new XGBoostClassifier().setNumRound(2)
    val transformedDf = classifier.fit(df).transform(df)

    // transform shouldn't change the features type
    val matched = transformedDf.schema("features").dataType match {
      case ArrayType(DoubleType, _) => true
      case _ => false
    }
    assert(matched)

    // No exception happened
    transformedDf.collect()
  }

  test("Fit with array and transform with vector type") {
    val df = smallBinaryClassificationArray
    val classifier = new XGBoostClassifier().setNumRound(2)
    val model = classifier.fit(df)

    val vectorDf = smallBinaryClassificationVector
    assert(SparkUtils.isVectorType(vectorDf.schema("features").dataType))

    val transformedDf = model.transform(vectorDf)
    assert(SparkUtils.isVectorType(transformedDf.schema("features").dataType))

    // No exception
    transformedDf.collect()
  }

  test("Fit with vector and transform with array type") {
    val vectorDf = smallBinaryClassificationVector

    val classifier = new XGBoostClassifier().setNumRound(2)
    val model = classifier.fit(vectorDf)

    val arrayDf = smallBinaryClassificationArray
    assert(classifier.featureIsArrayType(arrayDf.schema))

    val transformedDf = model.transform(arrayDf)
    assert(classifier.featureIsArrayType(transformedDf.schema))

    // No exception
    transformedDf.collect()
  }
}
