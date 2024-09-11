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

import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.xgboost.{SparkUtils, XGBProbabilisticClassifierParams}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}
import org.json4s.DefaultFormats

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.{BINARY_CLASSIFICATION_OBJS, MULTICLASSIFICATION_OBJS}

class XGBoostClassifier(override val uid: String,
                        private[spark] val xgboostParams: Map[String, Any])
  extends ProbabilisticClassifier[Vector, XGBoostClassifier, XGBoostClassificationModel]
    with XGBoostEstimator[XGBoostClassifier, XGBoostClassificationModel]
    with XGBProbabilisticClassifierParams[XGBoostClassifier] {

  def this() = this(XGBoostClassifier._uid, Map.empty)

  def this(uid: String) = this(uid, Map.empty)

  def this(xgboostParams: Map[String, Any]) = this(XGBoostClassifier._uid, xgboostParams)

  xgboost2SparkParams(xgboostParams)

  private var numberClasses = 0

  private def validateObjective(dataset: Dataset[_]): Unit = {
    // If the objective is set explicitly, it must be in BINARY_CLASSIFICATION_OBJS and
    // MULTICLASSIFICATION_OBJS
    val obj = if (isSet(objective)) {
      val tmpObj = getObjective
      val supportedObjs = BINARY_CLASSIFICATION_OBJS.toSeq ++ MULTICLASSIFICATION_OBJS.toSeq
      require(supportedObjs.contains(tmpObj),
        s"Wrong objective for XGBoostClassifier, supported objs: ${supportedObjs.mkString(",")}")
      Some(tmpObj)
    } else {
      None
    }

    def inferNumClasses: Int = {
      var num = getNumClass
      // Infer num class if num class is not set explicitly.
      // Note that user sets the num classes explicitly, we're not checking that.
      if (num == 0) {
        num = SparkUtils.getNumClasses(dataset, getLabelCol)
      }
      require(num > 0)
      num
    }

    // objective is set explicitly.
    if (obj.isDefined) {
      if (MULTICLASSIFICATION_OBJS.contains(getObjective)) {
        numberClasses = inferNumClasses
        setNumClass(numberClasses)
      } else {
        numberClasses = 2
        // binary classification doesn't require num_class be set
        require(!isSet(numClass), "num_class is not allowed for binary classification")
      }
    } else {
      // infer the objective according to the num_class
      numberClasses = inferNumClasses
      if (numberClasses <= 2) {
        setObjective("binary:logistic")
        logger.warn("Inferred for binary classification, set the objective to binary:logistic")
        require(!isSet(numClass), "num_class is not allowed for binary classification")
      } else {
        logger.warn("Inferred for multi classification, set the objective to multi:softprob")
        setObjective("multi:softprob")
        setNumClass(numberClasses)
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
    new XGBoostClassificationModel(uid, numberClasses, booster, Option(summary))
  }

}

object XGBoostClassifier extends DefaultParamsReadable[XGBoostClassifier] {
  private val _uid = Identifiable.randomUID("xgbc")
}

class XGBoostClassificationModel private[ml](
    val uid: String,
    val numClasses: Int,
    val nativeBooster: Booster,
    val summary: Option[XGBoostTrainingSummary] = None
) extends ProbabilisticClassificationModel[Vector, XGBoostClassificationModel]
  with XGBoostModel[XGBoostClassificationModel]
  with XGBProbabilisticClassifierParams[XGBoostClassificationModel] {

  def this(uid: String) = this(uid, 0, null)

  override protected[spark] def postTransform(dataset: Dataset[_],
                                              pred: PredictedColumns): Dataset[_] = {
    var output = super.postTransform(dataset, pred)

    // Always use probability col to get the prediction

    if (isDefinedNonEmpty(predictionCol) && pred.predTmp) {
      if (getObjective == "multi:softmax") {
        // For objective=multi:softmax scenario, there is no probability predicted from xgboost.
        // Instead, the probability column will be filled with real prediction
        val predictUDF = udf { probability: mutable.WrappedArray[Float] =>
          probability(0)
        }
        output = output.withColumn(getPredictionCol, predictUDF(col(TMP_TRANSFORMED_COL)))
      } else {
        val predCol = udf { probability: mutable.WrappedArray[Float] =>
          val prob = probability.map(_.toDouble).toArray
          val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
          probability2prediction(Vectors.dense(probabilities))
        }
        output = output.withColumn(getPredictionCol, predCol(col(TMP_TRANSFORMED_COL)))
      }
    }

    if (isDefinedNonEmpty(probabilityCol) && pred.predTmp) {
      val probabilityUDF = udf { probability: mutable.WrappedArray[Float] =>
        val prob = probability.map(_.toDouble).toArray
        val probabilities = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
        Vectors.dense(probabilities)
      }
      output = output.withColumn(TMP_TRANSFORMED_COL,
          probabilityUDF(output.col(TMP_TRANSFORMED_COL)))
        .withColumnRenamed(TMP_TRANSFORMED_COL, getProbabilityCol)
    }

    if (pred.predRaw) {
      val rawPredictionUDF = udf { raw: mutable.WrappedArray[Float] =>
        val rawF = raw.map(_.toDouble).toArray
        val rawPredictions = if (numClasses == 2) Array(-rawF(0), rawF(0)) else rawF
        Vectors.dense(rawPredictions)
      }
      output = output.withColumn(getRawPredictionCol,
        rawPredictionUDF(output.col(getRawPredictionCol)))
    }

    output.drop(TMP_TRANSFORMED_COL)
  }

  override def copy(extra: ParamMap): XGBoostClassificationModel = {
    val newModel = copyValues(new XGBoostClassificationModel(uid, numClasses,
      nativeBooster, summary), extra)
    newModel.setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    throw new Exception("XGBoost-Spark does not support \'raw2probabilityInPlace\'")
  }

  override def predictRaw(features: Vector): Vector =
    throw new Exception("XGBoost-Spark does not support \'predictRaw\'")

}

object XGBoostClassificationModel extends MLReadable[XGBoostClassificationModel] {

  override def read: MLReader[XGBoostClassificationModel] = new ModelReader

  private class ModelReader extends XGBoostModelReader[XGBoostClassificationModel] {
    override def load(path: String): XGBoostClassificationModel = {
      val xgbModel = loadBooster(path)
      val meta = SparkUtils.loadMetadata(path, sc)
      implicit val format = DefaultFormats
      val numClasses = (meta.params \ "numClass").extractOpt[Int].getOrElse(2)
      val model = new XGBoostClassificationModel(meta.uid, numClasses, xgbModel)
      meta.getAndSetParams(model)
      model
    }
  }
}
