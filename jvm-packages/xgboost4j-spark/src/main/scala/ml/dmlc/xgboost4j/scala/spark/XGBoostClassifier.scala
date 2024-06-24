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

import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.params.ClassificationParams
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.{binaryClassificationObjs, multiClassificationObjs}


class XGBoostClassifier(override val uid: String,
                        private[spark] val xgboostParams: Map[String, Any])
  extends XGBoostEstimator[XGBoostClassifier, XGBoostClassificationModel]
    with ClassificationParams[XGBoostClassifier] {

  def this() = this(Identifiable.randomUID("xgbc"), Map.empty)

  def this(uid: String) = this(uid, Map.empty)

  def this(xgboostParams: Map[String, Any]) = this(Identifiable.randomUID("xgbc"), xgboostParams)

  xgboost2SparkParams(xgboostParams)

  private def validateObjective(dataset: Dataset[_]): Unit = {
    // If the objective is set explicitly, it must be in binaryClassificationObjs and
    // multiClassificationObjs
    val obj = if (isSet(objective)) {
      val tmpObj = getObjective
      val supportedObjs = binaryClassificationObjs.toSeq ++ multiClassificationObjs.toSeq
      require(supportedObjs.contains(tmpObj),
        s"Wrong objective for XGBoostClassifier, supported objs: ${supportedObjs.mkString(",")}")
      Some(tmpObj)
    } else {
      None
    }

    def inferNumClasses: Int = {
      var numClasses = getNumClass
      // Infer num class if num class is not set explicitly.
      // Note that user sets the num classes explicitly, we're not checking that.
      if (numClasses == 0) {
        numClasses = SparkUtils.getNumClasses(dataset, getLabelCol)
      }
      require(numClasses > 0)
      numClasses
    }

    // objective is set explicitly.
    if (obj.isDefined) {
      if (multiClassificationObjs.contains(getObjective)) {
        setNumClass(inferNumClasses)
      } else {
        // binary classification doesn't require num_class be set
        require(!isSet(numClass), "num_class is not allowed for binary classification")
      }
    } else {
      // infer the objective according to the num_class
      val numClasses = inferNumClasses
      if (numClasses <= 2) {
        setObjective("binary:logistic")
        logger.warn("Inferred for binary classification, set the objective to binary:logistic")
        require(!isSet(numClass), "num_class is not allowed for binary classification")
      } else {
        logger.warn("Inferred for multi classification, set the objective to multi:softprob")
        setObjective("multi:softprob")
        setNumClass(numClasses)
      }
    }
  }

  /**
   * Validate the parameters before training, throw exception if possible
   */
  override protected[spark] def validate(dataset: Dataset[_]): Unit = {
    super.validate(dataset)
    validateObjective(dataset)
  }

  override protected def createModel(booster: Booster, summary: XGBoostTrainingSummary):
  XGBoostClassificationModel = {
    new XGBoostClassificationModel(uid, booster, Some(summary))
  }
}

object XGBoostClassifier extends DefaultParamsReadable[XGBoostClassifier] {
  private val _uid = Identifiable.randomUID("xgbc")

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
    // Always use probability col to get the prediction
    if (isDefinedNonEmpty(predictionCol)) {
      val predCol = udf { probability: mutable.WrappedArray[Float] =>
        probability2prediction(Vectors.dense(probability.map(_.toDouble).toArray))
      }
      output = output.withColumn(getPredictionCol, predCol(col(TMP_TRANSFORMED_COL)))
    }

    if (isDefinedNonEmpty(probabilityCol)) {
      output = output.withColumn(TMP_TRANSFORMED_COL,
          array_to_vector(output.col(TMP_TRANSFORMED_COL)))
        .withColumnRenamed(TMP_TRANSFORMED_COL, getProbabilityCol)
    }

    if (isDefinedNonEmpty(rawPredictionCol)) {
      output = output.withColumn(getRawPredictionCol,
        array_to_vector(output.col(getRawPredictionCol)))
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
