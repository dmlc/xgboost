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

import java.lang.reflect.Modifier

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
 * the estimator wrapping XGBoost to produce a training model
 *
 * @param xgboostParams the parameters configuring XGBoost
 */
class XGBoostEstimator private[spark](
  override val uid: String, xgboostParams: Map[String, Any])
  extends Predictor[MLVector, XGBoostEstimator, XGBoostModel]
  with LearningTaskParams with GeneralParams with BoosterParams {

  def this(xgboostParams: Map[String, Any]) =
    this(Identifiable.randomUID("XGBoostEstimator"), xgboostParams: Map[String, Any])

  def this(uid: String) = this(uid, Map[String, Any]())

  lazy val xgbParams: Array[Param[_]] = {
    val fields = classOf[BoosterParams].getDeclaredMethods ++
      classOf[GeneralParams].getDeclaredMethods ++ classOf[LearningTaskParams].getDeclaredMethods
    fields.filter { m => classOf[Param[_]].isAssignableFrom(m.getReturnType) &&
      m.getParameterTypes.isEmpty
    }.sortBy(_.getName).map(m => m.invoke(this).asInstanceOf[Param[_]])
  }

  // called in syncParams only when eval_metric is not defined
  private def deriveEvalMetric(): String = {
    val objFunc = xgboostParams.get("objective")
    if (objFunc.isEmpty) {
      "rmse"
    } else {
      // compute default metric based on specified objective
      val isClassificationTask = XGBoost.isClassificationTask(objFunc)
      if (!isClassificationTask) {
        // default metric for regression or ranking
        if (objFunc.get.toString.startsWith("rank")) {
          "map"
        } else {
          "rmse"
        }
      } else {
        // default metric for classification
        if (objFunc.get.toString.startsWith("multi")) {
          // multi
          "merror"
        } else {
          // binary
          "error"
        }
      }
    }
  }

  private def syncParams(): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      xgbParams.find(_.name == paramName) match {
        case None =>
        case Some(_: Param[_]) =>
          set(paramName, paramValue)
      }
    }
    if (xgboostParams.get("eval_metric").isEmpty) {
      set("eval_metric", deriveEvalMetric())
    }
  }

  syncParams()

  // only called when XGBParamMap is empty, i.e. in the constructor this(String)
  private def fromParamsToXGBParamMap(): Map[String, Any] = {
    require(xgboostParams.isEmpty, "fromParamsToXGBParamMap can only be called when" +
      " XGBParamMap is empty, i.e. in the constructor this(String)")
    val xgbParamMap = new mutable.HashMap[String, Any]()
    for (param <- xgbParams) {
      xgbParamMap += param.name -> $(param)
    }
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
    val localXGBParams = {
      if (xgboostParams.isEmpty) {
        fromParamsToXGBParamMap()
      } else {
        xgboostParams
      }
    }
    val trainedModel = XGBoost.trainWithRDD(instances, localXGBParams, $(round), $(nWorkers),
      $(customObj), $(customEval), $(useExternalMemory), $(missing)).setParent(this)
    val returnedModel = copyValues(trainedModel)
    if (XGBoost.isClassificationTask(
      if ($(customObj) == null) localXGBParams.get("objective") else
        localXGBParams.get("obj_type")))
    {
      val numClass = {
        if (localXGBParams.contains("num_class")) {
          localXGBParams("num_class").asInstanceOf[Int]
        } else {
          2
        }
      }
      returnedModel.asInstanceOf[XGBoostClassificationModel].numOfClasses = numClass
    }
    returnedModel
  }

  override def copy(extra: ParamMap): XGBoostEstimator = {
    defaultCopy(extra)
  }
}
