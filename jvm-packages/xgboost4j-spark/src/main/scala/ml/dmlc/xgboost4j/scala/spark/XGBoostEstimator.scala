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

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector => MLVector, VectorUDT}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, DoubleType}
import org.apache.spark.sql.{Dataset, Row}

/**
 * the estimator wrapping XGBoost to produce a training model
 *
 * @param xgboostParams the parameters configuring XGBoost
 * @param round the number of iterations to train
 * @param nWorkers the total number of workers of xgboost
 * @param obj the customized objective function, default to be null and using the default in model
 * @param eval the customized eval function, default to be null and using the default in model
 * @param useExternalMemory whether to use external memory when training
 * @param missing the value taken as missing
 */
class XGBoostEstimator private[spark](
    override val uid: String, xgboostParams: Map[String, Any], round: Int, nWorkers: Int,
    obj: ObjectiveTrait, eval: EvalTrait, useExternalMemory: Boolean, missing: Float)
  extends Predictor[MLVector, XGBoostEstimator, XGBoostModel] {

  def this(xgboostParams: Map[String, Any], round: Int, nWorkers: Int,
           obj: ObjectiveTrait = null,
           eval: EvalTrait = null, useExternalMemory: Boolean = false, missing: Float = Float.NaN) =
    this(Identifiable.randomUID("XGBoostEstimator"), xgboostParams: Map[String, Any], round: Int,
      nWorkers: Int, obj: ObjectiveTrait, eval: EvalTrait, useExternalMemory: Boolean,
      missing: Float)

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
    val trainedModel = XGBoost.trainWithRDD(instances, xgboostParams, round, nWorkers, obj,
      eval, useExternalMemory, missing).setParent(this)
    val returnedModel = copyValues(trainedModel)
    if (XGBoost.isClassificationTask(
      if (obj == null) xgboostParams.get("objective") else xgboostParams.get("obj_type"))) {
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
