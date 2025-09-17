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

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor._uid
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.REGRESSION_OBJS

class XGBoostRegressor(override val uid: String,
                       private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostEstimator[XGBoostRegressor, XGBoostRegressionModel] {

  def this() = this(_uid, Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(_uid, xgboostParams)

  xgboost2SparkParams(xgboostParams)

  /**
   * Validate the parameters before training, throw exception if possible
   */
  override protected[spark] def validate(dataset: Dataset[_]): Unit = {
    super.validate(dataset)

    // If the objective is set explicitly, it must be in REGRESSION_OBJS
    if (isSet(objective)) {
      val tmpObj = getObjective
      require(REGRESSION_OBJS.contains(tmpObj),
        s"Wrong objective for XGBoostRegressor, supported objs: ${REGRESSION_OBJS.mkString(",")}")
    }
  }

  override protected def createModel(
      booster: Booster,
      summary: XGBoostTrainingSummary): XGBoostRegressionModel = {
    new XGBoostRegressionModel(uid, booster, Option(summary))
  }

  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType =
    SparkUtils.appendColumn(schema, $(predictionCol), DoubleType)
}

object XGBoostRegressor extends DefaultParamsReadable[XGBoostRegressor] {
  private val _uid = Identifiable.randomUID("xgbr")
}

class XGBoostRegressionModel private[ml](val uid: String,
                                         val nativeBooster: Booster,
                                         val summary: Option[XGBoostTrainingSummary] = None)
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with RankerRegressorBaseModel[XGBoostRegressionModel] {

  def this(uid: String) = this(uid, null)

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    val newModel = copyValues(new XGBoostRegressionModel(uid, nativeBooster, summary), extra)
    newModel.setParent(parent)
  }

  override def predict(features: Vector): Double = {
    val values = predictSingleInstance(features)
    values(0)
  }
}

object XGBoostRegressionModel extends MLReadable[XGBoostRegressionModel] {
  override def read: MLReader[XGBoostRegressionModel] = new ModelReader

  private class ModelReader extends XGBoostModelReader[XGBoostRegressionModel] {
    override def load(path: String): XGBoostRegressionModel = {
      val xgbModel = loadBooster(path)
      val meta = SparkUtils.loadMetadata(path, sc)
      val model = new XGBoostRegressionModel(meta.uid, xgbModel, None)
      meta.getAndSetParams(model)
      model
    }
  }
}
