/*
 Copyright (c) 2014-2024 by Contributors

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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.spark.params.XGBoostParams

class XGBoostClassifierSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("params") {
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "objective" -> "binary:logistic"
    )
    val classifier = new XGBoostClassifier(xgbParams)
      .setFeaturesCol("abc")
      .setMissing(0.2f)
      .setAlpha(0.97)

    assert(classifier.getMaxDepth === 5)
    assert(classifier.getEta === 0.2)
    assert(classifier.getObjective === "binary:logistic")
    assert(classifier.getFeaturesCol === "abc")
    assert(classifier.getMissing === 0.2f)
    assert(classifier.getAlpha === 0.97)

    classifier.setEta(0.66).setMaxDepth(7)
    assert(classifier.getMaxDepth === 7)
    assert(classifier.getEta === 0.66)
  }

  test("XGBoostClassifier copy") {
    val classifier = new XGBoostClassifier().setNthread(2).setNumWorkers(10)
    val classifierCopied = classifier.copy(ParamMap.empty)

    assert(classifier.uid === classifierCopied.uid)
    assert(classifier.getNthread === classifierCopied.getNthread)
    assert(classifier.getNumWorkers === classifier.getNumWorkers)
  }

  test("XGBoostClassification copy") {
    val model = new XGBoostClassificationModel("hello").setNthread(2).setNumWorkers(10)
    val modelCopied = model.copy(ParamMap.empty)
    assert(model.uid === modelCopied.uid)
    assert(model.getNthread === modelCopied.getNthread)
    assert(model.getNumWorkers === modelCopied.getNumWorkers)
  }

  test("read/write") {
    val trainDf = smallBinaryClassificationVector
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "objective" -> "binary:logistic"
    )

    def check(xgboostParams: XGBoostParams[_]): Unit = {
      assert(xgboostParams.getMaxDepth === 5)
      assert(xgboostParams.getEta === 0.2)
      assert(xgboostParams.getObjective === "binary:logistic")
    }

    val classifierPath = new File(tempDir.toFile, "classifier").getPath
    val classifier = new XGBoostClassifier(xgbParams)
    check(classifier)

    classifier.write.overwrite().save(classifierPath)
    val loadedClassifier = XGBoostClassifier.load(classifierPath)
    check(loadedClassifier)

    val model = loadedClassifier.fit(trainDf)
    check(model)

    val modelPath = new File(tempDir.toFile, "model").getPath
    model.write.overwrite().save(modelPath)
    val modelLoaded = XGBoostClassificationModel.load(modelPath)
    check(modelLoaded)
  }


  test("pipeline") {
    val spark = ss
    var df = spark.read.parquet("/home/bobwang/data/iris/parquet")

    val conf = df.sparkSession.conf

    val x = conf.get("spark.rapids.sql.enabled", "false")
    println(x)


  }

  test("test NewXGBoostClassifierSuite") {
    // Define the schema for the fake data

    val spark = ss
    //        val features = Array("feature1", "feature2", "feature3", "feature4")

    //    val df = Seq(
    //      (1.0, 0.0, 0.0, 0.0, 0.0, 30),
    //      (2.0, 3.0, 4.0, 4.0, 0.0, 31),
    //      (3.0, 4.0, 5.0, 5.0, 1.0, 32),
    //      (4.0, 5.0, 6.0, 6.0, 1.0, 33),
    //    ).toDF("feature1", "feature2", "feature3", "feature4", "label", "base_margin")

    var df = spark.read.parquet("/home/bobwang/data/iris/parquet")

    // Select the features and label columns
    val labelCol = "class"

    val features = df.schema.names.filter(_ != labelCol)

    //    df = df.withColumn("base_margin", lit(20))
    //      .withColumn("weight", rand(1))

    // Assemble the feature columns into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")
    val dataset = assembler.transform(df)

    var Array(trainDf, validationDf) = dataset.randomSplit(Array(0.8, 0.2), seed = 1)

    //    trainDf = trainDf.withColumn("validation", lit(false))
    //    validationDf = validationDf.withColumn("validationDf", lit(true))

    //    df = trainDf.union(validationDf)

    //    val arrayInput = df.select(array(features.map(col(_)): _*).as("features"),
    //      col("label"), col("base_margin"))

    val est = new XGBoostClassifier()
      .setNumWorkers(1)
      .setNumRound(2)
      .setMaxDepth(3)
      //      .setWeightCol("weight")
      //      .setBaseMarginCol("base_margin")
      .setLabelCol(labelCol)
      .setEvalDataset(validationDf)
      //      .setValidationIndicatorCol("validation")
      //      .setPredictionCol("")
      .setRawPredictionCol("")
      .setProbabilityCol("xxxx")
    //      .setContribPredictionCol("contrb")
    //      .setLeafPredictionCol("leaf")
    //    val est = new XGBoostClassifier().setLabelCol(labelCol)
    //    est.fit(arrayInput)
    est.write.overwrite().save("/tmp/abcdef")
    val loadedEst = XGBoostClassifier.load("/tmp/abcdef")
    println(loadedEst.getNumRound)
    println(loadedEst.getMaxDepth)

    val model = est.fit(dataset)
    println("-----------------------")
    println(model.getNumRound)
    println(model.getMaxDepth)

    model.write.overwrite().save("/tmp/model/")
    val loadedModel = XGBoostClassificationModel.load("/tmp/model")
    println(loadedModel.getNumRound)
    println(loadedModel.getMaxDepth)
    model.transform(dataset).drop(features: _*).show(150, false)
  }

}
