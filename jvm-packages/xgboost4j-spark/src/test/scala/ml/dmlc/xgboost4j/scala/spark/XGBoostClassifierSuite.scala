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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.lit
import org.scalatest.funsuite.AnyFunSuite

class XGBoostClassifierSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

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
