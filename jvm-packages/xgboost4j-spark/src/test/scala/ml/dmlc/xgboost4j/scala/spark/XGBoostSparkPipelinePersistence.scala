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

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

class XGBoostSparkPipelinePersistence extends SharedSparkContext with Utils {
  test("test sparks pipeline persistence of dataframe-based model") {

    //  maybe move to shared context, but requires session to import implicits
    Logger.getLogger("org").setLevel(Level.WARN)

    val conf: SparkConf = new SparkConf()
      .setAppName("foo")
      .setMaster("local[*]")

    val spark: SparkSession = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    import spark.implicits._
    // maybe move to shared context, but requires session to import implicits

    val columnsFactor = Seq("bar", "baz")
    val columnsToDrop = Seq("dropme")

    val df = Seq((0, 0.5, 1, 0), (1, 0.01, 0.8, 9),
      (0, 0.8, 0.5, 6),
      (1, 8.4, 0.04, 4))
      .toDF("TARGET", "bar", "baz", "column")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(df.columns
        .filter(!_.contains("TARGET")))
      .setOutputCol("features")

    val xgbEstimator = new XGBoostEstimator(Map("num_rounds" -> 10))
      .setFeaturesCol("features")
      .setLabelCol("TARGET")

    val predictionModel = new Pipeline().setStages(Array(vectorAssembler, xgbEstimator)).fit(df)
    predictionModel.write.overwrite.save("testxgbPipe")
    val sameModel = XGBoostModel.load("testxgbPipe")
    assert(predictionModel == sameModel)
  }
}

