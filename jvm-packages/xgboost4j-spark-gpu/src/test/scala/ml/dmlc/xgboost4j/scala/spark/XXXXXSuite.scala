package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.lit
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.rapids.spark.GpuTestSuite

class XXXXXSuite extends AnyFunSuite with GpuTestSuite {

  test("test Gpu XGBoostClassifierSuite") {
    // Define the schema for the fake data

    withGpuSparkSession() { spark =>
      var df = spark.read.parquet("/home/bobwang/data/iris/parquet")

      df.sparkSession.conf.get("spark.rapids.sql.enabled")

      // Select the features and label columns
      val labelCol = "class"

      val features = df.schema.names.filter(_ != labelCol)

      //    df = df.withColumn("base_margin", lit(20))
      //      .withColumn("weight", rand(1))

      var Array(trainDf, validationDf) = df.randomSplit(Array(0.8, 0.2), seed = 1)

      trainDf = trainDf.withColumn("validation", lit(false))
      validationDf = validationDf.withColumn("validationDf", lit(true))

      df = trainDf.union(validationDf)
//
//      // Assemble the feature columns into a single vector column
//      val assembler = new VectorAssembler()
//        .setInputCols(features)
//        .setOutputCol("features")
//      val dataset = assembler.transform(df)

      //    val arrayInput = df.select(array(features.map(col(_)): _*).as("features"),
      //      col("label"), col("base_margin"))

      val est = new XGBoostClassifier()
        .setNumWorkers(1)
        .setNumRound(2)
        .setMaxDepth(3)
        //      .setWeightCol("weight")
        //      .setBaseMarginCol("base_margin")
        .setFeaturesCol(features)
        .setLabelCol(labelCol)
        .setValidationIndicatorCol("validation")
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

      val model = loadedEst.fit(df)
      println("-----------------------")
      println(model.getNumRound)
      println(model.getMaxDepth)

      model.write.overwrite().save("/tmp/model/")
      val loadedModel = XGBoostClassificationModel.load("/tmp/model")
      println(loadedModel.getNumRound)
      println(loadedModel.getMaxDepth)
      model.transform(df).drop(features: _*).show(150, false)
    }

  }
}
