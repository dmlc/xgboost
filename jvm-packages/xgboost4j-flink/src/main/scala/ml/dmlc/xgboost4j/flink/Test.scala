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

package ml.dmlc.xgboost4j.flink

import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.DataSet
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.MLUtils
import org.apache.flink.util.Collector



object Test {
  val log = LogFactory.getLog(this.getClass)
  def main(args: Array[String]) {
    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment
    val data = MLUtils.readLibSVM(env, "/home/tqchen/github/xgboost/demo/data/agaricus.txt.train")
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val round = 2
    val model = XGBoost.train(paramMap, data, round)


    log.info(model)
  }
}

