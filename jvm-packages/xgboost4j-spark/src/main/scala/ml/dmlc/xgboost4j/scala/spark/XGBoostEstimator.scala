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
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{VectorUDT, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{NumericType, DoubleType, StructType}
import org.apache.spark.sql.{Dataset, Row}

class XGBoostEstimator(
    inputCol: String, labelCol: String,
    xgboostParams: Map[String, Any], round: Int, nWorkers: Int,
    obj: ObjectiveTrait = null,
    eval: EvalTrait = null, useExternalMemory: Boolean = false, missing: Float = Float.NaN)
  extends Estimator[XGBoostModel] {

  override val uid: String = Identifiable.randomUID("XGBoostEstimator")

  override def fit(dataset: Dataset[_]): XGBoostModel = {
    val instances = dataset.select(col(inputCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(feature: Vector, label: Double) =>
        LabeledPoint(label, feature)
    }
    transformSchema(dataset.schema, logging = true)
    val trainedModel = XGBoost.trainWithRDD(instances, xgboostParams, round, nWorkers, obj,
      eval, useExternalMemory, missing).setParent(this)
    copyValues(trainedModel)
  }

  override def copy(extra: ParamMap): Estimator[XGBoostModel] = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    // check input type, for now we only support vectorUDT as the input feature type
    val inputType = schema(inputCol).dataType
    require(inputType.equals(new VectorUDT), s"the type of input column $inputCol has to VectorUDT")
    // check label Type,
    val labelType = schema(labelCol).dataType
    require(labelType.isInstanceOf[NumericType], s"the type of label column $labelCol has to" +
      s" be NumericType")
    schema
  }
}
