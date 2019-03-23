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

import ml.dmlc.xgboost4j.java.DistributedEvalError
import org.scalatest.FunSuite

class CustomizedEvalSuite extends FunSuite with PerTest {

  private val paramMap = List("eta" -> "1", "max_depth" -> "6",
    "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
    "custom_eval" -> new DistributedEvalError).toMap

  test("distributed training with customized evaluation metrics") {
    val trainingDF = buildDataFrame(Classification.train, numWorkers)
    val xgbModel = new XGBoostClassifier(paramMap).fit(trainingDF)

    println(xgbModel.summary.toString())
  }
}
