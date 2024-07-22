/*
 Copyright (c) 2024 by Contributors

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

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.Dataset

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoostRanker._uid
import ml.dmlc.xgboost4j.scala.spark.params.HasGroupCol
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.RANKER_OBJS

class XGBoostRanker(override val uid: String,
                    private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRanker, XGBoostRankerModel]
    with XGBoostEstimator[XGBoostRanker, XGBoostRankerModel] with HasGroupCol {

  def this() = this(_uid, Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(_uid, xgboostParams)

  def setGroupCol(value: String): XGBoostRanker = set(groupCol, value)

  xgboost2SparkParams(xgboostParams)

  /**
   * Validate the parameters before training, throw exception if possible
   */
  override protected[spark] def validate(dataset: Dataset[_]): Unit = {
    super.validate(dataset)

    // If the objective is set explicitly, it must be in binaryClassificationObjs and
    // multiClassificationObjs
    if (isSet(objective)) {
      val tmpObj = getObjective
      require(RANKER_OBJS.contains(tmpObj),
        s"Wrong objective for XGBoostRanker, supported objs: ${RANKER_OBJS.mkString(",")}")
    } else {
      setObjective("rank:ndcg")
    }

    require(isDefinedNonEmpty(groupCol), "groupCol needs to be set")
  }

  /**
   * Preprocess the dataset to meet the xgboost input requirement
   *
   * @param dataset
   * @return
   */
  override private[spark] def preprocess(dataset: Dataset[_]): (Dataset[_], ColumnIndices) = {
    val (output, columnIndices) = super.preprocess(dataset)
    (output.sortWithinPartitions(getGroupCol), columnIndices)
  }

  override protected def createModel(
      booster: Booster,
      summary: XGBoostTrainingSummary): XGBoostRankerModel = {
    new XGBoostRankerModel(uid, booster, Option(summary))
  }
}

object XGBoostRanker extends DefaultParamsReadable[XGBoostRanker] {
  private val _uid = Identifiable.randomUID("xgbranker")
}

class XGBoostRankerModel private[ml](val uid: String,
                                     val nativeBooster: Booster,
                                     val summary: Option[XGBoostTrainingSummary] = None)
  extends PredictionModel[Vector, XGBoostRankerModel]
    with RankerRegressorBaseModel[XGBoostRankerModel] with HasGroupCol {

  def this(uid: String) = this(uid, null)

  def setGroupCol(value: String): XGBoostRankerModel = set(groupCol, value)

  override def copy(extra: ParamMap): XGBoostRankerModel = {
    val newModel = copyValues(new XGBoostRankerModel(uid, nativeBooster, summary), extra)
    newModel.setParent(parent)
  }

  override def predict(features: Vector): Double = {
    val values = predictSingleInstance(features)
    values(0)
  }
}

object XGBoostRankerModel extends MLReadable[XGBoostRankerModel] {
  override def read: MLReader[XGBoostRankerModel] = new ModelReader

  private class ModelReader extends XGBoostModelReader[XGBoostRankerModel] {
    override def load(path: String): XGBoostRankerModel = {
      val xgbModel = loadBooster(path)
      val meta = SparkUtils.loadMetadata(path, sc)
      val model = new XGBoostRankerModel(meta.uid, xgbModel, None)
      meta.getAndSetParams(model)
      model
    }
  }
}
