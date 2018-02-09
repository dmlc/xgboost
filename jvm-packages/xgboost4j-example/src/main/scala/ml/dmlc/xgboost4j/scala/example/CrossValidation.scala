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
package ml.dmlc.xgboost4j.scala.example

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}

object CrossValidation {
  def main(args: Array[String]): Unit = {
    val trainMat: DMatrix = new DMatrix("../../demo/data/agaricus.txt.train")

    // set params
    val params = new mutable.HashMap[String, Any]

    params.put("eta", 1.0)
    params.put("max_depth", 3)
    params.put("silent", 1)
    params.put("nthread", 6)
    params.put("objective", "binary:logistic")
    params.put("gamma", 1.0)
    params.put("eval_metric", "error")

    // do 5-fold cross validation
    val round: Int = 2
    val nfold: Int = 5
    // set additional eval_metrics
    val metrics: Array[String] = null

    val evalHist: Array[String] =
      XGBoost.crossValidation(trainMat, params.toMap, round, nfold, metrics)
  }
}
