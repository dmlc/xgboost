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

import ml.dmlc.xgboost4j.XGBoostError
import org.apache.commons.logging.LogFactory
import org.scalatest.FunSuite

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

  test("basic operation of booster") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test")

    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val watches = List("train" -> trainMat, "test" -> testMat).toMap

    val round = 2
    val booster = XGBoost.train(paramMap, trainMat, round, watches, null, null)
    val predicts = booster.predict(testMat, true)
    val eval = new EvalError
    assert(eval.eval(predicts, testMat) < 0.1)
  }

  test("cross validation") {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train")
    val params = List("eta" -> "1.0", "max_depth" -> "3", "slient" -> "1", "nthread" -> "6",
      "objective" -> "binary:logistic", "gamma" -> "1.0", "eval_metric" -> "error").toMap
    val round = 2
    val nfold = 5
    XGBoost.crossValiation(params, trainMat, round, nfold, null, null, null)
  }
}
