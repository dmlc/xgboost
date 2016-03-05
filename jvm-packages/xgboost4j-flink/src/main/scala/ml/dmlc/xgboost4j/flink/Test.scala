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

import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker}
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.XGBoost
import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.DataSet
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.MLUtils
import org.apache.flink.util.Collector

class ScalaMapFunction(workerEnvs: java.util.Map[String, String])
  extends RichMapPartitionFunction[LabeledVector, Booster] {
  val log = LogFactory.getLog(this.getClass)
  def mapPartition(it : java.lang.Iterable[LabeledVector], collector: Collector[Booster]): Unit = {
    workerEnvs.put("DMLC_TASK_ID", String.valueOf(this.getRuntimeContext.getIndexOfThisSubtask))
    log.info("start with env" + workerEnvs.toString)
    Rabit.init(workerEnvs)

    val trainMat = new DMatrix("/home/tqchen/github/xgboost/demo/data/agaricus.txt.train")

    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val watches = List("train" -> trainMat).toMap
    val round = 2
    val booster = XGBoost.train(paramMap, trainMat, round, watches, null, null)
    Rabit.shutdown()
    collector.collect(booster)
  }
}



object Test {
  val log = LogFactory.getLog(this.getClass)
  def main(args: Array[String]) {
    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment
    val data = MLUtils.readLibSVM(env, "/home/tqchen/github/xgboost/demo/data/agaricus.txt.train")
    val tracker = new RabitTracker(data.getExecutionEnvironment.getParallelism)
    log.info("start with parallelism" + data.getExecutionEnvironment.getParallelism)
    assert(data.getExecutionEnvironment.getParallelism >= 1)
    tracker.start()

    val res = data.mapPartition(new ScalaMapFunction(tracker.getWorkerEnvs)).reduce((x, y) => x)
    val model = res.collect()
    log.info(model)
  }
}

