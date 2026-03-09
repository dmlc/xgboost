/*
 Copyright (c) 2014-2026 by Contributors

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

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}

object PredictLeafIndices {

  def main(args: Array[String]): Unit = {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train?format=libsvm")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test?format=libsvm")

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 1.0
    params += "max_depth" -> 2
    params += "silent" -> 1
    params += "objective" -> "binary:logistic"

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMat
    watches += "test" -> testMat

    val round = 3
    val booster = XGBoost.train(trainMat, params.toMap, round, watches.toMap)

    // predict using first 2 tree
    val leafIndex = booster.predictLeaf(testMat, 2)
    if (leafIndex.length > 0 && leafIndex(0).length > 1) {
      println(s"${leafIndex(0)(0)}, ${leafIndex(0)(1)}")
    }

    // predict all trees
    val leafIndex2 = booster.predictLeaf(testMat, 0)
    if (leafIndex2.length > 0 && leafIndex2(0).length > 1) {
      println(s"${leafIndex2(0)(0)}, ${leafIndex2(0)(1)}")
    }
  }
}
