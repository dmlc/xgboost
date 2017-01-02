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

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

case class Foobar(TARGET: Int, bar: Double, baz: Double)

class XGBoostSparkPipelinePersistence extends SharedSparkContext with Utils {
  test("test sparks pipeline persistence of dataframe-based model") {
    //  maybe move to shared context, but requires session to import implicits.
    // what about introducing https://github.com/holdenk/spark-testing-base ?
    val conf: SparkConf = new SparkConf()
      .setAppName("foo")
      .setMaster("local[*]")

    val spark: SparkSession = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    import spark.implicits._
    // maybe move to shared context, but requires session to import implicits

    val df = Seq(Foobar(0, 0.5, 1), Foobar(1, 0.01, 0.8),
      Foobar(0, 0.8, 0.5), Foobar(1, 8.4, 0.04))
      .toDS

    val vectorAssembler = new VectorAssembler()
      .setInputCols(df.columns
        .filter(!_.contains("TARGET")))
      .setOutputCol("features")

    val xgbEstimator = new XGBoostEstimator(Map("num_rounds" -> 10))
      .setFeaturesCol("features")
      .setLabelCol("TARGET")

    // separate
    val predModel = xgbEstimator.fit(vectorAssembler.transform(df))
    predModel.write.overwrite.save("test2xgbPipe")
    val same2Model = XGBoostModel.load("test2xgbPipe")

    assert(java.util.Arrays.equals(predModel.booster.toByteArray, same2Model.booster.toByteArray))
    // cant directly compare parameters as parent (object id) is used in equals.
    // only want to compre the value
    // but where to get thw value? Spark internal
    // $(same2Model.useExternalMemory) does not work
    // and this neither predModel.useExternalMemory.getOrDefault("use_external_memory")
    // do not only want to compare names but rather values ...

    assert(predModel.useExternalMemory.name == same2Model.useExternalMemory.name)
    assert(predModel.useExternalMemory == same2Model.useExternalMemory)
    assert(predModel.featuresCol == same2Model.featuresCol)
    assert(predModel.predictionCol == same2Model.predictionCol)
    assert(predModel.labelCol == same2Model.labelCol)

    //    val memoryPredictions = predModel.transform(vectorAssembler.transform(df))
    //    memoryPredictions.show
    //    val loadedPredictions = same2Model.transform(vectorAssembler.transform(df))
    //    // TODO this doesn't work -> will compare prediction results like other test cases
    //    // assert(predModel === same2Model)
    //    assert(loadedPredictions.collect === memoryPredictions.collect)
    //
    //    // chained
    val predictionModel = new Pipeline().setStages(Array(vectorAssembler, xgbEstimator)).fit(df)
    predictionModel.write.overwrite.save("testxgbPipe")
    val sameModel = PipelineModel.load("testxgbPipe")

    val predictionModelXGB = predictionModel.stages collect { case xgb: XGBoostModel => xgb } head
    val sameModelXGB = predictionModel.stages collect { case xgb: XGBoostModel => xgb } head

    assert(java.util.Arrays.equals(
      predictionModelXGB.booster.toByteArray,
      sameModelXGB.booster.toByteArray
    ))
    assert(predictionModelXGB.useExternalMemory == sameModelXGB.useExternalMemory)
    assert(predictionModelXGB.featuresCol == sameModelXGB.featuresCol)
    assert(predictionModelXGB.predictionCol == sameModelXGB.predictionCol)
    assert(predictionModelXGB.labelCol == sameModelXGB.labelCol)
    //
    //    val memoryPredictionsPipe = predictionModel.transform(df)
    //    val loadedPredictionsPipe = sameModel.transform(df)
    //    assert(memoryPredictionsPipe.collect === loadedPredictionsPipe.collect)
  }
}

