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

package ml.dmlc.xgboost4j.scala.spark

import scala.util.Try

import org.scalatest.funsuite.AnyFunSuite


class XGBoostParamsSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("invalid parameters") {
    val estimator = new XGBoostClassifier()

    // We didn't set it by default
    var thrown = intercept[RuntimeException] {
      estimator.getCacheHostRatio
    }
    assert(thrown.getMessage.contains("Failed to find a default value for cacheHostRatio"))

    val v = Try(estimator.getCacheHostRatio).getOrElse(Float.NaN)
    assert(v.equals(Float.NaN))

    // We didn't set it by default
    thrown = intercept[RuntimeException] {
      estimator.setCacheHostRatio(-1.0f)
    }
    assert(thrown.getMessage.contains("parameter cacheHostRatio given invalid value -1.0"))

    Seq(0.0f, 0.2f, 1.0f).forall(v => {
      estimator.setCacheHostRatio(v)
      estimator.getCacheHostRatio == v
    })

    estimator.setCacheHostRatio(0.66f)
    val v1 = Try(estimator.getCacheHostRatio).getOrElse(Float.NaN)
    assert(v1 == 0.66f)
  }

  test("setNumEarlyStoppingRounds") {
    val estimator = new XGBoostClassifier()
    assert(estimator.getNumEarlyStoppingRounds == 0)
    estimator.setNumEarlyStoppingRounds(10)
    assert(estimator.getNumEarlyStoppingRounds == 10)
  }

}
