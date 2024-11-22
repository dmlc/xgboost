/*
 Copyright (c) 2024 by Contributors

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

import org.scalatest.funsuite.AnyFunSuite

class ScalaExamplesTest extends AnyFunSuite {
  test("Smoke test for Scala examples") {
    val args = Array("")
    println("BasicWalkThrough")
    BasicWalkThrough.main(args)
    println("BoostFromPrediction")
    BoostFromPrediction.main(args)
    println("CrossValidation")
    CrossValidation.main(args)
    println("CustomObjective")
    CustomObjective.main(args)
    println("ExternalMemory")
    ExternalMemory.main(args)
    println("GeneralizedLinearModel")
    GeneralizedLinearModel.main(args)
    println("PredictFirstNTree")
    PredictFirstNTree.main(args)
    println("PredictLeafIndices")
    PredictLeafIndices.main(args)
  }
}
