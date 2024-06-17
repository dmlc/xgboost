/*
 Copyright (c) 2014-2024 by Contributors

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

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader, SchemaUtils}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.params.ClassificationParams


class XGBoostClassifier(override val uid: String,
                        private[spark] val xgboostParams: Map[String, Any])
  extends XGBoostEstimator[XGBoostClassifier, XGBoostClassificationModel]
    with ClassificationParams[XGBoostClassifier] {

  def this() = this(Identifiable.randomUID("xgbc"), Map.empty)

  def this(uid: String) = this(uid, Map.empty)

  def this(xgboostParams: Map[String, Any]) = this(Identifiable.randomUID("xgbc"), xgboostParams)

  xgboost2SparkParams(xgboostParams)

  /**
   * Validate the parameters before training, throw exception if possible
   */
  override protected def validate(dataset: Dataset[_]): Unit = {
    super.validate(dataset)

    // The default objective is for regression case.
    val obj = if (isSet(objective)) {
      Some(getObjective)
    } else {
      None
    }

    var numClasses = getNumClass
    // If user didn't set it, inferred it.
    if (numClasses == 0) {
      numClasses = SparkUtils.getNumClasses(dataset, getLabelCol)
    }
    assert(numClasses > 0)

    if (numClasses <= 2) {
      if (!obj.exists(_.startsWith("binary:"))) {
        logger.warn(s"Inferred for binary classification, but found wrong objective: " +
          s"${getObjective}, rewrite objective to binary:logistic")
        setObjective("binary:logistic")
      }
    } else {
      if (!obj.exists(_.startsWith("multi:"))) {
        logger.warn(s"Inferred for multiclass classification, but found wrong objective: " +
          s"${getObjective}, rewrite objective to multi:softprob")
        setObjective("multi:softprob")
      }
      setNumClass(numClasses)
    }

  }

  override protected def createModel(booster: Booster, summary: XGBoostTrainingSummary):
  XGBoostClassificationModel = {
    new XGBoostClassificationModel(uid, booster, Some(summary))
  }
}

object XGBoostClassifier extends DefaultParamsReadable[XGBoostClassifier] {
  private val uid = Identifiable.randomUID("xgbc")
  override def load(path: String): XGBoostClassifier = super.load(path)
}

// TODO add num classes
class XGBoostClassificationModel(
                                  uid: String,
                                  model: Booster,
                                  trainingSummary: Option[XGBoostTrainingSummary] = None
                                )
  extends XGBoostModel[XGBoostClassificationModel](uid, model, trainingSummary)
    with ClassificationParams[XGBoostClassificationModel] {

  def this(uid: String) = this(uid, null)

  // Copied from Spark
  private def probability2prediction(probability: Vector): Double = {
    if (!isDefined(thresholds)) {
      probability.argmax
    } else {
      val thresholds = getThresholds
      var argMax = 0
      var max = Double.NegativeInfinity
      var i = 0
      val probabilitySize = probability.size
      while (i < probabilitySize) {
        // Thresholds are all > 0, excepting that at most one may be 0.
        // The single class whose threshold is 0, if any, will always be predicted
        // ('scaled' = +Infinity). However in the case that this class also has
        // 0 probability, the class will not be selected ('scaled' is NaN).
        val scaled = probability(i) / thresholds(i)
        if (scaled > max) {
          max = scaled
          argMax = i
        }
        i += 1
      }
      argMax
    }
  }

  override def postTransform(dataset: Dataset[_]): Dataset[_] = {
    var output = dataset
    if (isDefined(predictionCol) && getPredictionCol.nonEmpty) {
      val predCol = udf { probability: mutable.WrappedArray[Float] =>
        probability2prediction(Vectors.dense(probability.map(_.toDouble).toArray))
      }
      output = output.withColumn(getPredictionCol, predCol(col(TMP_TRANSFORMED_COL)))
    }

    if (isDefined(probabilityCol) && getProbabilityCol.nonEmpty) {
      output = output.withColumnRenamed(TMP_TRANSFORMED_COL, getProbabilityCol)
    }
    output.drop(TMP_TRANSFORMED_COL)
  }

  override def copy(extra: ParamMap): XGBoostClassificationModel = {
    val newModel = copyValues(new XGBoostClassificationModel(uid, model, trainingSummary), extra)
    newModel.setParent(parent)
  }
}

object XGBoostClassificationModel extends MLReadable[XGBoostClassificationModel] {

  override def read: MLReader[XGBoostClassificationModel] = new ModelReader

  private class ModelReader extends XGBoostModelReader[XGBoostClassificationModel] {
    override def load(path: String): XGBoostClassificationModel = {
      val xgbModel = loadBooster(path)
      val meta = SparkUtils.loadMetadata(path, sc)
      val model = new XGBoostClassificationModel(meta.uid, xgbModel)
      meta.getAndSetParams(model)
      model
    }
  }
}
