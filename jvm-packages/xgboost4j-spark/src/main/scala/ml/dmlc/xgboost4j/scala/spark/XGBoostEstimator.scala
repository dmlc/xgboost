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

import ml.dmlc.xgboost4j.scala.spark.params._
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

import org.apache.spark.ml.Predictor
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.FloatType
import org.apache.spark.sql.{Dataset, Row}
import org.json4s.DefaultFormats

/**
 * XGBoost Estimator to produce a XGBoost model
 */
class XGBoostEstimator private[spark](
  override val uid: String, xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostEstimator, XGBoostModel]
  with LearningTaskParams with GeneralParams with BoosterParams with MLWritable {

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

  private[spark] def fromParamsToXGBParamMap: Map[String, Any] = {
    val xgbParamMap = new mutable.HashMap[String, Any]()
    for (param <- params) {
      xgbParamMap += param.name -> $(param)
    }
    val r = xgbParamMap.toMap
    if (!XGBoost.isClassificationTask(r) || $(numClasses) == 2) {
      r - "num_class"
    } else {
      r
    }
  }

  private def ensureColumns(trainingSet: Dataset[_]): Dataset[_] = {
    var newTrainingSet = trainingSet
    if (!trainingSet.columns.contains($(baseMarginCol))) {
      newTrainingSet = newTrainingSet.withColumn($(baseMarginCol), lit(Float.NaN))
    }
    if (!trainingSet.columns.contains($(weightCol))) {
      newTrainingSet = newTrainingSet.withColumn($(weightCol), lit(1.0))
    }
    newTrainingSet
  }

  /**
   * produce a XGBoostModel by fitting the given dataset
   */
  override def train(trainingSet: Dataset[_]): XGBoostModel = {
    val instances = ensureColumns(trainingSet).select(
      col($(featuresCol)),
      col($(labelCol)).cast(FloatType),
      col($(baseMarginCol)).cast(FloatType),
      col($(weightCol)).cast(FloatType)
    ).rdd.map { case Row(features: Vector, label: Float, baseMargin: Float, weight: Float) =>
      val (indices, values) = features match {
        case v: SparseVector => (v.indices, v.values.map(_.toFloat))
        case v: DenseVector => (null, v.values.map(_.toFloat))
      }
      XGBLabeledPoint(label.toFloat, indices, values, baseMargin = baseMargin, weight = weight)
    }
    transformSchema(trainingSet.schema, logging = true)
    val derivedXGBoosterParamMap = fromParamsToXGBParamMap
    val trainedModel = XGBoost.trainDistributed(instances, derivedXGBoosterParamMap,
      $(round), $(nWorkers), $(customObj), $(customEval), $(useExternalMemory),
      $(missing)).setParent(this)
    val returnedModel = copyValues(trainedModel, extractParamMap())
    if (XGBoost.isClassificationTask(derivedXGBoosterParamMap)) {
      returnedModel.asInstanceOf[XGBoostClassificationModel].numOfClasses = $(numClasses)
    }
    returnedModel
  }

  override def copy(extra: ParamMap): XGBoostEstimator = {
    defaultCopy(extra).asInstanceOf[XGBoostEstimator]
  }

  override def write: MLWriter = new XGBoostEstimator.XGBoostEstimatorWriter(this)
}

object XGBoostEstimator extends MLReadable[XGBoostEstimator] {

  override def read: MLReader[XGBoostEstimator] = new XGBoostEstimatorReader

  override def load(path: String): XGBoostEstimator = super.load(path)

  private[XGBoostEstimator] class XGBoostEstimatorWriter(instance: XGBoostEstimator)
    extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      require(instance.fromParamsToXGBParamMap("custom_eval") == null &&
        instance.fromParamsToXGBParamMap("custom_obj") == null,
        "we do not support persist XGBoostEstimator with customized evaluator and objective" +
          " function for now")
      implicit val format = DefaultFormats
      implicit val sc = super.sparkSession.sparkContext
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
    }
  }

  private class XGBoostEstimatorReader extends MLReader[XGBoostEstimator] {

    override def load(path: String): XGBoostEstimator = {
      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
      val cls = Utils.classForName(metadata.className)
      val instance =
        cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
      DefaultXGBoostParamsReader.getAndSetParams(instance, metadata)
      instance.asInstanceOf[XGBoostEstimator]
    }
  }
}
