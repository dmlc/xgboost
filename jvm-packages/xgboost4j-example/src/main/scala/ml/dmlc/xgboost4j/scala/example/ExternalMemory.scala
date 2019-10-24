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

object ExternalMemory {
  def main(args: Array[String]): Unit = {
    // this is the only difference, add a # followed by a cache prefix name
    // several cache file with the prefix will be generated
    // currently only support convert from libsvm file
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train#dtrain.cache")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test#dtest.cache")

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 1.0
    params += "max_depth" -> 2
    params += "silent" -> 1
    params += "objective" -> "binary:logistic"

    // performance notice: set nthread to be the number of your real cpu
    // some cpu offer two threads per core, for example, a 4 core cpu with 8 threads, in such case
    // set nthread=4
    // param.put("nthread", num_real_cpu);

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMat
    watches += "test" -> testMat

    val round = 2
    // train a model
    val booster = XGBoost.train(trainMat, params.toMap, round, watches.toMap)

    val trainPred = booster.predict(trainMat, true)
    val testPred = booster.predict(testMat, true)

    trainMat.setBaseMargin(trainPred)
    testMat.setBaseMargin(testPred)

    System.out.println("result of running from initial prediction")
    val booster2 = XGBoost.train(trainMat, params.toMap, 1, watches.toMap)
  }
}
