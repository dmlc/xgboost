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

import java.io.File
import java.nio.file.Files

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}

class CheckpointManagerSuite extends FunSuite  with BeforeAndAfterAll {
  var sc: SparkContext = _

  override def beforeAll(): Unit = {
    val conf: SparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("XGBoostSuite")
    sc = new SparkContext(conf)
  }

  private lazy val (model4, model8) = {
    import DataUtils._
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML).cache()
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    (XGBoost.trainWithRDD(trainingRDD, paramMap, round = 2, sc.defaultParallelism),
      XGBoost.trainWithRDD(trainingRDD, paramMap, round = 4, sc.defaultParallelism))
  }

  test("test update/load models") {
    val tmpPath = Files.createTempDirectory("test").toAbsolutePath.toString
    val manager = new CheckpointManager(sc, tmpPath)
    manager.updateCheckpoint(model4)
    var files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
    assert(files.length == 1)
    assert(files.head.getPath.getName == "4.model")
    assert(manager.loadCheckpointAsBooster.booster.getVersion == 4)

    manager.updateCheckpoint(model8)
    files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
    assert(files.length == 1)
    assert(files.head.getPath.getName == "8.model")
    assert(manager.loadCheckpointAsBooster.booster.getVersion == 8)
  }

  test("test cleanUpHigherVersions") {
    val tmpPath = Files.createTempDirectory("test").toAbsolutePath.toString
    val manager = new CheckpointManager(sc, tmpPath)
    manager.updateCheckpoint(model8)
    manager.cleanUpHigherVersions(round = 8)
    assert(new File(s"$tmpPath/8.model").exists())

    manager.cleanUpHigherVersions(round = 4)
    assert(!new File(s"$tmpPath/8.model").exists())
  }

  test("test checkpoint rounds") {
    val tmpPath = Files.createTempDirectory("test").toAbsolutePath.toString
    val manager = new CheckpointManager(sc, tmpPath)
    assertResult(Seq(7))(manager.getCheckpointRounds(checkpointInterval = 0, round = 7))
    assertResult(Seq(2, 4, 6, 7))(manager.getCheckpointRounds(checkpointInterval = 2, round = 7))
    manager.updateCheckpoint(model4)
    assertResult(Seq(4, 6, 7))(manager.getCheckpointRounds(2, 7))
  }
}
