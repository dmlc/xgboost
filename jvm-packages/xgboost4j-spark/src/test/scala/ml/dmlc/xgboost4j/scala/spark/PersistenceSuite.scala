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

import java.io.{File, FileNotFoundException}
import java.util.Arrays

import ml.dmlc.xgboost4j.scala.DMatrix

import scala.util.Random
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.network.util.JavaUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class PersistenceSuite extends FunSuite with PerTest with BeforeAndAfterAll {

  private var tempDir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    tempDir = new File(System.getProperty("java.io.tmpdir"), this.getClass.getName)
    if (tempDir.exists) {
      tempDir.delete
    }
    tempDir.mkdirs
  }

  override def afterAll(): Unit = {
    JavaUtils.deleteRecursively(tempDir)
    super.afterAll()
  }

  private def delete(f: File) {
    if (f.exists) {
      if (f.isDirectory) {
        for (c <- f.listFiles) {
          delete(c)
        }
      }
      if (!f.delete) {
        throw new FileNotFoundException("Failed to delete file: " + f)
      }
    }
  }

  test("test persistence of XGBoostClassifier and XGBoostClassificationModel") {
    val eval = new EvalError()
    val trainingDF = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)

    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> "10", "num_workers" -> numWorkers)
    val xgbc = new XGBoostClassifier(paramMap)
    val xgbcPath = new File(tempDir, "xgbc").getPath
    xgbc.write.overwrite().save(xgbcPath)
    val xgbc2 = XGBoostClassifier.load(xgbcPath)
    val paramMap2 = xgbc2.MLlib2XGBoostParams
    paramMap.foreach {
      case (k, v) => assert(v.toString == paramMap2(k).toString)
    }

    val model = xgbc.fit(trainingDF)
    val evalResults = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults < 0.1)
    val xgbcModelPath = new File(tempDir, "xgbcModel").getPath
    model.write.overwrite.save(xgbcModelPath)
    val model2 = XGBoostClassificationModel.load(xgbcModelPath)
    assert(Arrays.equals(model._booster.toByteArray, model2._booster.toByteArray))

    assert(model.getEta === model2.getEta)
    assert(model.getNumRound === model2.getNumRound)
    assert(model.getRawPredictionCol === model2.getRawPredictionCol)
    val evalResults2 = eval.eval(model2._booster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults === evalResults2)
  }

  test("test persistence of XGBoostRegressor and XGBoostRegressionModel") {
    val eval = new EvalError()
    val trainingDF = buildDataFrame(Regression.train)
    val testDM = new DMatrix(Regression.test.iterator)

    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:squarederror", "num_round" -> "10", "num_workers" -> numWorkers)
    val xgbr = new XGBoostRegressor(paramMap)
    val xgbrPath = new File(tempDir, "xgbr").getPath
    xgbr.write.overwrite().save(xgbrPath)
    val xgbr2 = XGBoostRegressor.load(xgbrPath)
    val paramMap2 = xgbr2.MLlib2XGBoostParams
    paramMap.foreach {
      case (k, v) => assert(v.toString == paramMap2(k).toString)
    }

    val model = xgbr.fit(trainingDF)
    val evalResults = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults < 0.1)
    val xgbrModelPath = new File(tempDir, "xgbrModel").getPath
    model.write.overwrite.save(xgbrModelPath)
    val model2 = XGBoostRegressionModel.load(xgbrModelPath)
    assert(Arrays.equals(model._booster.toByteArray, model2._booster.toByteArray))

    assert(model.getEta === model2.getEta)
    assert(model.getNumRound === model2.getNumRound)
    assert(model.getPredictionCol === model2.getPredictionCol)
    val evalResults2 = eval.eval(model2._booster.predict(testDM, outPutMargin = true), testDM)
    assert(evalResults === evalResults2)
  }

  test("test persistence of MLlib pipeline with XGBoostClassificationModel") {

    val r = new Random(0)
    // maybe move to shared context, but requires session to import implicits
    val df = ss.createDataFrame(Seq.fill(100)(r.nextInt(2)).map(i => (i, i))).
      toDF("feature", "label")

    val assembler = new VectorAssembler()
      .setInputCols(df.columns.filter(!_.contains("label")))
      .setOutputCol("features")

    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> "10", "num_workers" -> numWorkers)
    val xgb = new XGBoostClassifier(paramMap)

    // Construct MLlib pipeline, save and load
    val pipeline = new Pipeline().setStages(Array(assembler, xgb))
    val pipePath = new File(tempDir, "pipeline").getPath
    pipeline.write.overwrite().save(pipePath)
    val pipeline2 = Pipeline.read.load(pipePath)
    val xgb2 = pipeline2.getStages(1).asInstanceOf[XGBoostClassifier]
    val paramMap2 = xgb2.MLlib2XGBoostParams
    paramMap.foreach {
      case (k, v) => assert(v.toString == paramMap2(k).toString)
    }

    // Model training, save and load
    val pipeModel = pipeline.fit(df)
    val pipeModelPath = new File(tempDir, "pipelineModel").getPath
    pipeModel.write.overwrite.save(pipeModelPath)
    val pipeModel2 = PipelineModel.load(pipeModelPath)

    val xgbModel = pipeModel.stages(1).asInstanceOf[XGBoostClassificationModel]
    val xgbModel2 = pipeModel2.stages(1).asInstanceOf[XGBoostClassificationModel]

    assert(Arrays.equals(xgbModel._booster.toByteArray, xgbModel2._booster.toByteArray))

    assert(xgbModel.getEta === xgbModel2.getEta)
    assert(xgbModel.getNumRound === xgbModel2.getNumRound)
    assert(xgbModel.getRawPredictionCol === xgbModel2.getRawPredictionCol)
  }
}

