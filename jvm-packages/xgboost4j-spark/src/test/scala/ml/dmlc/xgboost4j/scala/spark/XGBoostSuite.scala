/*
 Copyright (c) 2023-2024 by Contributors

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

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.Booster

class XGBoostSuite extends AnyFunSuite with PerTest {

  // Do not create spark context
  override def beforeEach(): Unit = {}

  test("skip stage-level scheduling") {
    val conf = new SparkConf()
      .setMaster("spark://foo")
      .set("spark.executor.cores", "12")
      .set("spark.task.cpus", "1")
      .set("spark.executor.resource.gpu.amount", "1")
      .set("spark.task.resource.gpu.amount", "0.08")

    // the correct configurations should not skip stage-level scheduling
    assert(!XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, conf))

    // spark version < 3.4.0
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.3.0", runOnGpu = true, conf))

    // not run on GPU
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = false, conf))

    // spark.executor.cores is not set
    var badConf = conf.clone().remove("spark.executor.cores")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // spark.executor.cores=1
    badConf = conf.clone().set("spark.executor.cores", "1")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // spark.executor.resource.gpu.amount is not set
    badConf = conf.clone().remove("spark.executor.resource.gpu.amount")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // spark.executor.resource.gpu.amount>1
    badConf = conf.clone().set("spark.executor.resource.gpu.amount", "2")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // spark.task.resource.gpu.amount is not set
    badConf = conf.clone().remove("spark.task.resource.gpu.amount")
    assert(!XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // spark.task.resource.gpu.amount=1
    badConf = conf.clone().set("spark.task.resource.gpu.amount", "1")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // yarn
    badConf = conf.clone().setMaster("yarn")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))

    // k8s
    badConf = conf.clone().setMaster("k8s://")
    assert(XGBoost.skipStageLevelScheduling(sparkVersion = "3.4.0", runOnGpu = true, badConf))
  }


  object FakedXGBoost extends StageLevelScheduling {

    // Do not skip stage-level scheduling for testing purposes.
    override private[spark] def skipStageLevelScheduling(
        sparkVersion: String,
        runOnGpu: Boolean,
        conf: SparkConf) = false
  }

  test("try stage-level scheduling without spark-rapids") {

    val builder = SparkSession.builder()
      .master(s"local-cluster[1, 4, 1024]")
      .appName("XGBoostSuite")
      .config("spark.ui.enabled", false)
      .config("spark.driver.memory", "512m")
      .config("spark.barrier.sync.timeout", 10)
      .config("spark.task.cpus", 1)
      .config("spark.executor.cores", 4)
      .config("spark.executor.resource.gpu.amount", 1)
      .config("spark.task.resource.gpu.amount", 0.25)
    val ss = builder.getOrCreate()
    if (ss.version < "3.4.1") {
      // Pass
      ss.stop()
    } else {
      try {
        val df = ss.range(1, 10)
        val rdd = df.rdd

        val runtimeParams = new XGBoostClassifier(
          Map("device" -> "cuda")).setNumWorkers(1).setNumRound(1)
          .getRuntimeParameters(true)
        assert(runtimeParams.runOnGpu)

        val finalRDD = FakedXGBoost.tryStageLevelScheduling(ss.sparkContext, runtimeParams,
          rdd.asInstanceOf[RDD[(Booster, Map[String, Array[Float]])]])

        val taskResources = finalRDD.getResourceProfile().taskResources
        assert(taskResources.contains("cpus"))
        assert(taskResources.get("cpus").get.amount == 3)

        assert(taskResources.contains("gpu"))
        assert(taskResources.get("gpu").get.amount == 1.0)
      } finally {
        ss.stop()
      }
    }
  }
}
