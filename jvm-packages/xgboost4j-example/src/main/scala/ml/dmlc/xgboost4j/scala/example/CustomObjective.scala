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
package ml.dmlc.xgboost4j.scala.example

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix, EvalTrait, ObjectiveTrait}
import org.apache.commons.logging.{LogFactory, Log}

/**
 * an example user define objective and eval
 * NOTE: when you do customized loss function, the default prediction value is margin
 * this may make buildin evalution metric not function properly
 * for example, we are doing logistic loss, the prediction is score before logistic transformation
 * he buildin evaluation error assumes input is after logistic transformation
 * Take this in mind when you use the customization, and maybe you need write customized evaluation
 * function
 *
 */
object CustomObjective {

  /**
   * loglikelihoode loss obj function
   */
  class LogRegObj extends ObjectiveTrait {
    private val logger: Log = LogFactory.getLog(classOf[LogRegObj])
    /**
     * user define objective function, return gradient and second order gradient
     *
     * @param predicts untransformed margin predicts
     * @param dtrain   training data
     * @return List with two float array, correspond to first order grad and second order grad
     */
    override def getGradient(predicts: Array[Array[Float]], dtrain: DMatrix)
        : List[Array[Float]] = {
      val nrow = predicts.length
      val gradients = new ListBuffer[Array[Float]]
      var labels: Array[Float] = null
      try {
        labels = dtrain.getLabel
      } catch {
        case e: XGBoostError =>
          logger.error(e)
          null
        case _: Throwable =>
          null
      }
      val grad = new Array[Float](nrow)
      val hess = new Array[Float](nrow)
      val transPredicts = transform(predicts)

      for (i <- 0 until nrow) {
        val predict = transPredicts(i)(0)
        grad(i) = predict - labels(i)
        hess(i) = predict * (1 - predict)
      }
      gradients += grad
      gradients += hess
      gradients.toList
    }

    /**
     * simple sigmoid func
     *
     * @param input
     * @return Note: this func is not concern about numerical stability, only used as example
     */
    def sigmoid(input: Float): Float = {
      (1 / (1 + Math.exp(-input))).toFloat
    }

    def transform(predicts: Array[Array[Float]]): Array[Array[Float]] = {
      val nrow = predicts.length
      val transPredicts = Array.fill[Float](nrow, 1)(0)
      for (i <- 0 until nrow) {
        transPredicts(i)(0) = sigmoid(predicts(i)(0))
      }
      transPredicts
    }

  }

  class EvalError extends EvalTrait {

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

  def main(args: Array[String]): Unit = {
    val trainMat = new DMatrix("../../demo/data/agaricus.txt.train?format=libsvm")
    val testMat = new DMatrix("../../demo/data/agaricus.txt.test?format=libsvm")
    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 1.0
    params += "max_depth" -> 2
    params += "silent" -> 1
    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMat
    watches += "test" -> testMat

    val round = 2
    // train a model
    val booster = XGBoost.train(trainMat, params.toMap, round, watches.toMap)
    XGBoost.train(trainMat, params.toMap, round, watches.toMap,
      obj = new LogRegObj, eval = new EvalError)
  }

}
