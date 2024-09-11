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

package org.apache.spark.ml.xgboost

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.ProbabilisticClassifierParams
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DatasetUtils, DefaultParamsReader, DefaultParamsWriter, SchemaUtils}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.json4s.{JObject, JValue}

import ml.dmlc.xgboost4j.scala.spark.params.NonXGBoostParams

/**
 * XGBoost classification spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be XGBoostClassifier or XGBoostClassificationModel
 */
trait XGBProbabilisticClassifierParams[T <: Params]
  extends ProbabilisticClassifierParams with NonXGBoostParams {

  /**
   * XGBoost doesn't use validateAndTransformSchema since spark validateAndTransformSchema
   * needs to ensure the feature is vector type
   */
  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    var outputSchema = SparkUtils.appendColumn(schema, $(predictionCol), DoubleType)
    outputSchema = SparkUtils.appendVectorUDTColumn(outputSchema, $(rawPredictionCol))
    outputSchema = SparkUtils.appendVectorUDTColumn(outputSchema, $(probabilityCol))
    outputSchema
  }

  addNonXGBoostParam(rawPredictionCol, probabilityCol, thresholds)
}

/** Utils to access the spark internal functions */
object SparkUtils {

  def getNumClasses(dataset: Dataset[_], labelCol: String, maxNumClasses: Int = 100): Int = {
    DatasetUtils.getNumClasses(dataset, labelCol, maxNumClasses)
  }

  def checkNumericType(schema: StructType, colName: String, msg: String = ""): Unit = {
    SchemaUtils.checkNumericType(schema, colName, msg)
  }

  def saveMetadata(instance: Params,
                   path: String,
                   sc: SparkContext,
                   extraMetadata: Option[JObject] = None,
                   paramMap: Option[JValue] = None): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, paramMap)
  }

  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
  }

  def appendColumn(schema: StructType,
                   colName: String,
                   dataType: DataType,
                   nullable: Boolean = false): StructType = {
    SchemaUtils.appendColumn(schema, colName, dataType, nullable)
  }

  def appendVectorUDTColumn(schema: StructType,
                            colName: String,
                            dataType: DataType = new VectorUDT,
                            nullable: Boolean = false): StructType = {
    SchemaUtils.appendColumn(schema, colName, dataType, nullable)
  }
}
