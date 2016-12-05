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

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.spark.params.{BoosterParams, GeneralParams, LearningTaskParams}
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}

/**
 * XGBoost Estimator to produce a XGBoost model
 */
class XGBoostEstimator private[spark](
  override val uid: String, private[spark] var xgboostParams: Map[String, Any])
  extends Predictor[MLVector, XGBoostEstimator, XGBoostModel]
  with LearningTaskParams with GeneralParams with BoosterParams {

  def this(xgboostParams: Map[String, Any]) =
    this(Identifiable.randomUID("XGBoostEstimator"), xgboostParams: Map[String, Any])

  def this(uid: String) = this(uid, Map[String, Any]())


  // called in fromXGBParamMapToParams only when eval_metric is not defined
  private def setupDefaultEvalMetric(): String = {
    val objFunc = xgboostParams.getOrElse("objective", xgboostParams.getOrElse("obj_type", null))
    if (objFunc == null) {
      "rmse"
    } else {
      // compute default metric based on specified objective
      val isClassificationTask = XGBoost.isClassificationTask(xgboostParams)
      if (!isClassificationTask) {
        // default metric for regression or ranking
        if (objFunc.toString.startsWith("rank")) {
          "map"
        } else {
          "rmse"
        }
      } else {
        // default metric for classification
        if (objFunc.toString.startsWith("multi")) {
          // multi
          "merror"
        } else {
          // binary
          "error"
        }
      }
    }
  }

  private def fromXGBParamMapToParams(): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      params.find(_.name == paramName) match {
        case None =>
        case Some(_: DoubleParam) =>
          set(paramName, paramValue.toString.toDouble)
        case Some(_: BooleanParam) =>
          set(paramName, paramValue.toString.toBoolean)
        case Some(_: IntParam) =>
          set(paramName, paramValue.toString.toInt)
        case Some(_: FloatParam) =>
          set(paramName, paramValue.toString.toFloat)
        case Some(_: Param[_]) =>
          set(paramName, paramValue)
      }
    }
    if (xgboostParams.get("eval_metric").isEmpty) {
      set("eval_metric", setupDefaultEvalMetric())
    }
  }

  fromXGBParamMapToParams()

  // only called when XGBParamMap is empty, i.e. in the constructor this(String)
  // TODO: refactor to be functional
  private def fromParamsToXGBParamMap(): Map[String, Any] = {
    require(xgboostParams.isEmpty, "fromParamsToXGBParamMap can only be called when" +
      " XGBParamMap is empty, i.e. in the constructor this(String)")
    val xgbParamMap = new mutable.HashMap[String, Any]()
    for (param <- params) {
      xgbParamMap += param.name -> $(param)
    }
    xgboostParams = xgbParamMap.toMap
    xgbParamMap.toMap
  }

  /**
   * produce a XGBoostModel by fitting the given dataset
   */
  override def train(trainingSet: Dataset[_]): XGBoostModel = {
    val instances = trainingSet.select(
      col($(featuresCol)), col($(labelCol)).cast(DoubleType)).rdd.map {
      case Row(feature: MLVector, label: Double) =>
        LabeledPoint(label, feature)
    }
    transformSchema(trainingSet.schema, logging = true)
    val trainedModel = XGBoost.trainWithRDD(instances, xgboostParams, $(round), $(nWorkers),
      $(customObj), $(customEval), $(useExternalMemory), $(missing)).setParent(this)
    val returnedModel = copyValues(trainedModel)
    if (XGBoost.isClassificationTask(xgboostParams)) {
      val numClass = {
        if (xgboostParams.contains("num_class")) {
          xgboostParams("num_class").asInstanceOf[Int]
        } else {
          2
        }
      }
      returnedModel.asInstanceOf[XGBoostClassificationModel].numOfClasses = numClass
    }
    returnedModel
  }

  override def copy(extra: ParamMap): XGBoostEstimator = {
    val est = defaultCopy(extra).asInstanceOf[XGBoostEstimator]
    // we need to synchronize the params here instead of in the constructor
    // because we cannot guarantee that params (default implementation) is initialized fully
    // before the other params
    est.fromParamsToXGBParamMap()
    est
  }
}
