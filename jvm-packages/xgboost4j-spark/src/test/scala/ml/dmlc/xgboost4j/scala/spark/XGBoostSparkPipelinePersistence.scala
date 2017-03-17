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

import scala.concurrent.duration._

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

    val xgbEstimator = new XGBoostEstimator(Map("num_rounds" -> 10,
      "tracker_conf" -> TrackerConf(1 minute, "scala")
    ))
      .setFeaturesCol("features")
      .setLabelCol("TARGET")

    // separate
    val predModel = xgbEstimator.fit(vectorAssembler.transform(df))
    predModel.write.overwrite.save("test2xgbPipe")
    val same2Model = XGBoostModel.load("test2xgbPipe")

    assert(java.util.Arrays.equals(predModel.booster.toByteArray, same2Model.booster.toByteArray))
    val predParamMap = predModel.extractParamMap()
    val same2ParamMap = same2Model.extractParamMap()
    assert(predParamMap.get(predModel.useExternalMemory)
      === same2ParamMap.get(same2Model.useExternalMemory))
    assert(predParamMap.get(predModel.featuresCol) === same2ParamMap.get(same2Model.featuresCol))
    assert(predParamMap.get(predModel.predictionCol)
      === same2ParamMap.get(same2Model.predictionCol))
    assert(predParamMap.get(predModel.labelCol) === same2ParamMap.get(same2Model.labelCol))
    assert(predParamMap.get(predModel.labelCol) === same2ParamMap.get(same2Model.labelCol))

    // chained
    val predictionModel = new Pipeline().setStages(Array(vectorAssembler, xgbEstimator)).fit(df)
    predictionModel.write.overwrite.save("testxgbPipe")
    val sameModel = PipelineModel.load("testxgbPipe")

    val predictionModelXGB = predictionModel.stages.collect { case xgb: XGBoostModel => xgb } head
    val sameModelXGB = sameModel.stages.collect { case xgb: XGBoostModel => xgb } head

    assert(java.util.Arrays.equals(
      predictionModelXGB.booster.toByteArray,
      sameModelXGB.booster.toByteArray
    ))
    val predictionModelXGBParamMap = predictionModel.extractParamMap()
    val sameModelXGBParamMap = sameModel.extractParamMap()
    assert(predictionModelXGBParamMap.get(predictionModelXGB.useExternalMemory)
      === sameModelXGBParamMap.get(sameModelXGB.useExternalMemory))
    assert(predictionModelXGBParamMap.get(predictionModelXGB.featuresCol)
      === sameModelXGBParamMap.get(sameModelXGB.featuresCol))
    assert(predictionModelXGBParamMap.get(predictionModelXGB.predictionCol)
      === sameModelXGBParamMap.get(sameModelXGB.predictionCol))
    assert(predictionModelXGBParamMap.get(predictionModelXGB.labelCol)
      === sameModelXGBParamMap.get(sameModelXGB.labelCol))
    assert(predictionModelXGBParamMap.get(predictionModelXGB.labelCol)
      === sameModelXGBParamMap.get(sameModelXGB.labelCol))
  }
}

