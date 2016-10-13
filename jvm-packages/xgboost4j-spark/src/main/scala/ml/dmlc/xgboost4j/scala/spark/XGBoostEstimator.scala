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

import ml.dmlc.xgboost4j.scala.spark.params.{BoosterParams, GeneralParams, LearningTaskParams}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector => MLVector, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, DoubleType}
import org.apache.spark.sql.{Dataset, Row}

/**
 * the estimator wrapping XGBoost to produce a training model
 *
 * @param xgboostParams the parameters configuring XGBoost
 * @param missing the value taken as missing
 */
class XGBoostEstimator private[spark](
  override val uid: String, xgboostParams: Map[String, Any], missing: Float)
  extends Predictor[MLVector, XGBoostEstimator, XGBoostModel]
  with LearningTaskParams with GeneralParams with BoosterParams {

  private[spark] def this(xgboostParams: Map[String, Any], missing: Float = Float.NaN) =
    this(Identifiable.randomUID("XGBoostEstimator"), xgboostParams: Map[String, Any],
      missing: Float)

  private def syncParams(): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      $(params.find(_.name == paramName).get) match {
        case dp: Double =>
          set(paramName, paramValue.toString.toDouble)
        case ip: Int =>
          set(paramName, paramValue.toString.toInt)
        case bp: Boolean =>
          set(paramName, bp.toString.toBoolean)
        case sp: String =>
          set(paramName, paramValue.toString)
        case _ =>
          // pass
      }
    }
  }

  syncParams()

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
      $(customObj), $(customEval), $(useExternalMemory), missing).setParent(this)
    val returnedModel = copyValues(trainedModel)
    if (XGBoost.isClassificationTask(
      if ($(customObj) == null) xgboostParams.get("objective") else xgboostParams.get("obj_type")))
    {
      val numClass = {
        if (xgboostParams.contains("num_class")) {
          xgboostParams("num_class").asInstanceOf[Int]
        }
        else {
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
