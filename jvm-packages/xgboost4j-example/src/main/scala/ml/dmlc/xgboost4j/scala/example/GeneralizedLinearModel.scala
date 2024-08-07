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
package ml.dmlc.xgboost4j.scala.example

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import ml.dmlc.xgboost4j.scala.example.util.CustomEval


/**
 * this is an example of fit generalized linear model in xgboost
 * basically, we are using linear model, instead of tree for our boosters
 */
object GeneralizedLinearModel {
  def main(args: Array[String]): Unit = {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train?format=libsvm")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test?format=libsvm")

    // specify parameters
    // change booster to gblinear, so that we are fitting a linear model
    // alpha is the L1 regularizer
    // lambda is the L2 regularizer
    // you can also set lambda_bias which is L2 regularizer on the bias term
    val params = new mutable.HashMap[String, Any]()
    params += "alpha" -> 0.0001
    params += "boosterh" -> "gblinear"
    params += "silent" -> 1
    params += "objective" -> "binary:logistic"

    // normally, you do not need to set eta (step_size)
    // XGBoost uses a parallel coordinate descent algorithm (shotgun),
    // there could be affection on convergence with parallelization on certain cases
    // setting eta to be smaller value, e.g 0.5 can make the optimization more stable
    // param.put("eta", "0.5");

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMat
    watches += "test" -> testMat

    val booster = XGBoost.train(trainMat, params.toMap, 1, watches.toMap)
    val predicts = booster.predict(testMat)
    val eval = new CustomEval
    println(s"error=${eval.eval(predicts, testMat)}")
  }
}
