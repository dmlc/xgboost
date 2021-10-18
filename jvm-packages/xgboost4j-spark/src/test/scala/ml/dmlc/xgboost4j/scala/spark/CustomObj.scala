/*
 Copyright (c) 2021 by Contributors

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

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.{DMatrix, ObjectiveTrait}
import org.apache.commons.logging.LogFactory
import scala.collection.mutable.ListBuffer


/**
 * loglikelihood loss obj function
 */
class CustomObj(val customParameter: Int = 0) extends ObjectiveTrait {

  val logger = LogFactory.getLog(classOf[CustomObj])

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
        throw e
      case e: Throwable => throw e
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
