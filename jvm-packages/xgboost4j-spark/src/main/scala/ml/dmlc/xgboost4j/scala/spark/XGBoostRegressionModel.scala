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

import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}

class XGBoostRegressionModel private[spark](override val uid: String, _booster: Booster)
  extends XGBoostModel(_booster) {

  def this(_booster: Booster) = this(Identifiable.randomUID("XGBoostRegressionModel"), _booster)

  override protected def transformImpl(testSet: Dataset[_]): DataFrame = {
    transformSchema(testSet.schema, logging = true)
    val predictRDD = produceRowRDD(testSet)
    testSet.sparkSession.createDataFrame(predictRDD, schema =
      StructType(testSet.schema.add(StructField($(predictionCol),
        ArrayType(FloatType, containsNull = false), nullable = false)))
    )
  }

  override protected def predict(features: MLVector): Double = {
    throw new Exception("XGBoost does not support online prediction for now")
  }

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    defaultCopy(extra)
  }
}
