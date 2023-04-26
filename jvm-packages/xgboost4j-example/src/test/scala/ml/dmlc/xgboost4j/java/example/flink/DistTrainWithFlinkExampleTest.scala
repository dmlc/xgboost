/*
 Copyright (c) 2014-2023 by Contributors

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
package ml.dmlc.xgboost4j.java.example.flink

import org.apache.flink.api.java.ExecutionEnvironment
import org.scalatest.Inspectors._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import java.nio.file.Paths

class DistTrainWithFlinkExampleTest extends AnyFunSuite {
  private val parentPath = Paths.get("../../").resolve("demo").resolve("data")
  private val data = parentPath.resolve("veterans_lung_cancer.csv")

  test("Smoke test for scala flink example") {
    val env = ExecutionEnvironment.createLocalEnvironment(1)
    val tuple2 = DistTrainWithFlinkExample.runPrediction(env, data, 70)
    val results = tuple2.f1.collect()
    results should have size 41
    forEvery(results)(item => item should have size 1)
  }
}
