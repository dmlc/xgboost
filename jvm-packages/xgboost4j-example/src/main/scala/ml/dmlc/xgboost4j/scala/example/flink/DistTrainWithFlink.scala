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
package ml.dmlc.xgboost4j.scala.example.flink

import ml.dmlc.xgboost4j.scala.flink.XGBoost
import org.apache.flink.api.scala.{ExecutionEnvironment, _}
import org.apache.flink.ml.MLUtils

object DistTrainWithFlink {
  def main(args: Array[String]) {
    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment
    // read trainining data
    val trainData =
      MLUtils.readLibSVM(env, "/path/to/data/agaricus.txt.train")
    val testData = MLUtils.readLibSVM(env, "/path/to/data/agaricus.txt.test")
    // define parameters
    val paramMap = List(
      "eta" -> 0.1,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    // number of iterations
    val round = 2
    // train the model
    val model = XGBoost.train(trainData, paramMap, round)
    val predTest = model.predict(testData.map{x => x.vector})
    model.saveModelAsHadoopFile("file:///path/to/xgboost.model")
  }
}
