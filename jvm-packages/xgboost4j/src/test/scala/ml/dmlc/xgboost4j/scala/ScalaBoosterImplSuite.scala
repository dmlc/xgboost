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

package ml.dmlc.xgboost4j.scala

import java.io.{FileOutputStream, FileInputStream, File}

import junit.framework.TestCase
import org.apache.commons.logging.LogFactory
import org.scalatest.FunSuite

import ml.dmlc.xgboost4j.java.XGBoostError

class ScalaBoosterImplSuite extends FunSuite {

  private class EvalError extends EvalTrait {

    val logger = LogFactory.getLog(classOf[EvalError])

    private[xgboost4j] var evalMetric: String = "custom_error"

    /**
     * get evaluate metric
     *
     * @return evalMetric
     */
    override def getMetric: String = evalMetric

    /**
     * evaluate with predicts and data
     *
     * @param predicts predictions as array
     * @param dmat     data matrix to evaluate
     * @return result of the metric
     */
    override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = {
      var error: Float = 0f
      var labels: Array[Float] = null
      try {
        labels = dmat.getLabel
      } catch {
        case ex: XGBoostError =>
          logger.error(ex)
          return -1f
      }
      val nrow: Int = predicts.length
      for (i <- 0 until nrow) {
        if (labels(i) == 0.0 && predicts(i)(0) > 0) {
          error += 1
        } else if (labels(i) == 1.0 && predicts(i)(0) <= 0) {
          error += 1
        }
      }
      error / labels.length
    }
  }

  private def trainBooster(trainMat: DMatrix, testMat: DMatrix): Booster = {
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val watches = List("train" -> trainMat, "test" -> testMat).toMap

    val round = 2
    XGBoost.train(trainMat, paramMap, round, watches)
  }

  private def trainBoosterWithQuantileHisto(
      trainMat: DMatrix,
      watches: Map[String, DMatrix],
      round: Int,
      paramMap: Map[String, String],
      threshold: Float): Booster = {
    val metrics = Array.fill(watches.size, round)(0.0f)
    val booster = XGBoost.train(trainMat, paramMap, round, watches, metrics)
    for (i <- 0 until watches.size; j <- 1 until metrics(i).length) {
      assert(metrics(i)(j) >= metrics(i)(j - 1))
    }
    for (metricsArray <- metrics; m <- metricsArray) {
      assert(m >= threshold)
    }
    booster
  }

  test("basic operation of booster") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")

    val booster = trainBooster(trainMat, testMat)
    val predicts = booster.predict(testMat, true)
    val eval = new EvalError
    assert(eval.eval(predicts, testMat) < 0.1)
  }

  test("save/load model with path") {

    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val eval = new EvalError
    val booster = trainBooster(trainMat, testMat)
    // save and load
    val temp: File = File.createTempFile("temp", "model")
    temp.deleteOnExit()
    booster.saveModel(temp.getAbsolutePath)

    val bst2: Booster = XGBoost.loadModel(temp.getAbsolutePath)
    assert(java.util.Arrays.equals(bst2.toByteArray, booster.toByteArray))
    val predicts2: Array[Array[Float]] = bst2.predict(testMat, true, 0)
    TestCase.assertTrue(eval.eval(predicts2, testMat) < 0.1f)
  }

  test("save/load model with stream") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val eval = new EvalError
    val booster = trainBooster(trainMat, testMat)
    // save and load
    val temp: File = File.createTempFile("temp", "model")
    temp.deleteOnExit()
    booster.saveModel(new FileOutputStream(temp.getAbsolutePath))

    val bst2: Booster = XGBoost.loadModel(new FileInputStream(temp.getAbsolutePath))
    assert(java.util.Arrays.equals(bst2.toByteArray, booster.toByteArray))
    val predicts2: Array[Array[Float]] = bst2.predict(testMat, true, 0)
    TestCase.assertTrue(eval.eval(predicts2, testMat) < 0.1f)
  }

  test("cross validation") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val params = List("eta" -> "1.0", "max_depth" -> "3", "silent" -> "1", "nthread" -> "6",
      "objective" -> "binary:logistic", "gamma" -> "1.0", "eval_metric" -> "error").toMap
    val round = 2
    val nfold = 5
    XGBoost.crossValidation(trainMat, params, round, nfold)
  }

  test("test with quantile histo depthwise") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val paramMap = List("max_depth" -> "3", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "eval_metric" -> "auc").toMap
    trainBoosterWithQuantileHisto(trainMat, Map("training" -> trainMat, "test" -> testMat),
      round = 10, paramMap, 0.95f)
  }

  test("test with quantile histo lossguide") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val paramMap = List("max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "lossguide", "max_leaves" -> "8", "eval_metric" -> "auc").toMap
    trainBoosterWithQuantileHisto(trainMat, Map("training" -> trainMat, "test" -> testMat),
      round = 10, paramMap, 0.95f)
  }

  test("test with quantile histo lossguide with max bin") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val paramMap = List("max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "lossguide", "max_leaves" -> "8", "max_bin" -> "16",
      "eval_metric" -> "auc").toMap
    trainBoosterWithQuantileHisto(trainMat, Map("training" -> trainMat),
      round = 10, paramMap, 0.95f)
  }

  test("test with quantile histo depthwidth with max depth") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val paramMap = List("max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_leaves" -> "8", "max_depth" -> "2",
      "eval_metric" -> "auc").toMap
    trainBoosterWithQuantileHisto(trainMat, Map("training" -> trainMat),
      round = 10, paramMap, 0.95f)
  }

  test("test with quantile histo depthwidth with max depth and max bin") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")
    val paramMap = List("max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
      "eval_metric" -> "auc").toMap
    trainBoosterWithQuantileHisto(trainMat, Map("training" -> trainMat),
      round = 10, paramMap, 0.95f)
  }

  test("test training from existing model in scala") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val paramMap = List("max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
      "eval_metric" -> "auc").toMap

    val prevBooster = XGBoost.train(trainMat, paramMap, round = 2)
    val nextBooster = XGBoost.train(trainMat, paramMap, round = 4, booster = prevBooster)
    assert(prevBooster == nextBooster)
  }
}
