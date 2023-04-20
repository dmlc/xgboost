/*
 Copyright (c) 2014-2022 by Contributors

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

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, ExternalCheckpointManager, XGBoost => SXGBoost}
import org.scalatest.funsuite.AnyFunSuite
import org.apache.hadoop.fs.{FileSystem, Path}

class ExternalCheckpointManagerSuite extends AnyFunSuite with TmpFolderPerSuite with PerTest {

  private def produceParamMap(checkpointPath: String, checkpointInterval: Int):
  Map[String, Any] = {
    Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic", "num_workers" -> sc.defaultParallelism,
      "checkpoint_path" -> checkpointPath, "checkpoint_interval" -> checkpointInterval)
  }

  private def createNewModels():
    (String, XGBoostClassificationModel, XGBoostClassificationModel) = {
    val tmpPath = createTmpFolder("test").toAbsolutePath.toString
    val (model4, model8) = {
      val training = buildDataFrame(Classification.train)
      val paramMap = produceParamMap(tmpPath, 2)
      (new XGBoostClassifier(paramMap ++ Seq("num_round" -> 2)).fit(training),
        new XGBoostClassifier(paramMap ++ Seq("num_round" -> 4)).fit(training))
    }
    (tmpPath, model4, model8)
  }

  test("test update/load models") {
    val (tmpPath, model4, model8) = createNewModels()
    val manager = new ExternalCheckpointManager(tmpPath, FileSystem.get(sc.hadoopConfiguration))

    manager.updateCheckpoint(model4._booster.booster)
    var files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
    assert(files.length == 1)
    assert(files.head.getPath.getName == "4.model")
    assert(manager.loadCheckpointAsScalaBooster().getVersion == 4)

    manager.updateCheckpoint(model8._booster)
    files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
    assert(files.length == 1)
    assert(files.head.getPath.getName == "8.model")
    assert(manager.loadCheckpointAsScalaBooster().getVersion == 8)
  }

  test("test cleanUpHigherVersions") {
    val (tmpPath, model4, model8) = createNewModels()

    val manager = new ExternalCheckpointManager(tmpPath, FileSystem.get(sc.hadoopConfiguration))
    manager.updateCheckpoint(model8._booster)
    manager.cleanUpHigherVersions(8)
    assert(new File(s"$tmpPath/8.model").exists())

    manager.cleanUpHigherVersions(4)
    assert(!new File(s"$tmpPath/8.model").exists())
  }

  test("test checkpoint rounds") {
    import scala.collection.JavaConverters._
    val (tmpPath, model4, model8) = createNewModels()
    val manager = new ExternalCheckpointManager(tmpPath, FileSystem.get(sc.hadoopConfiguration))
    assertResult(Seq(7))(
      manager.getCheckpointRounds(0, 7).asScala)
    assertResult(Seq(2, 4, 6, 7))(
      manager.getCheckpointRounds(2, 7).asScala)
    manager.updateCheckpoint(model4._booster)
    assertResult(Seq(4, 6, 7))(
      manager.getCheckpointRounds(2, 7).asScala)
  }


  private def trainingWithCheckpoint(cacheData: Boolean, skipCleanCheckpoint: Boolean): Unit = {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)

    val tmpPath = createTmpFolder("model1").toAbsolutePath.toString

    val paramMap = produceParamMap(tmpPath, 2)

    val cacheDataMap = if (cacheData) Map("cacheTrainingSet" -> true) else Map()
    val skipCleanCheckpointMap =
      if (skipCleanCheckpoint) Map("skip_clean_checkpoint" -> true) else Map()

    val finalParamMap = paramMap ++ cacheDataMap ++ skipCleanCheckpointMap

    val prevModel = new XGBoostClassifier(finalParamMap ++ Seq("num_round" -> 5)).fit(training)

    def error(model: Booster): Float = eval.eval(model.predict(testDM, outPutMargin = true), testDM)

    if (skipCleanCheckpoint) {
      // Check only one model is kept after training
      val files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
      assert(files.length == 1)
      assert(files.head.getPath.getName == "8.model")
      val tmpModel = SXGBoost.loadModel(s"$tmpPath/8.model")
      // Train next model based on prev model
      val nextModel = new XGBoostClassifier(paramMap ++ Seq("num_round" -> 8)).fit(training)
      assert(error(tmpModel) >= error(prevModel._booster))
      assert(error(prevModel._booster) > error(nextModel._booster))
      assert(error(nextModel._booster) < 0.1)
    } else {
      assert(!FileSystem.get(sc.hadoopConfiguration).exists(new Path(tmpPath)))
    }
  }

  test("training with checkpoint boosters") {
    trainingWithCheckpoint(cacheData = false, skipCleanCheckpoint = true)
  }

  test("training with checkpoint boosters with cached training dataset") {
    trainingWithCheckpoint(cacheData = true, skipCleanCheckpoint = true)
  }

  test("the checkpoint file should be cleaned after a successful training") {
    trainingWithCheckpoint(cacheData = false, skipCleanCheckpoint = false)
  }
}
