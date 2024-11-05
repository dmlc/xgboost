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

import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.ml.classification.ProbabilisticClassifierParams
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.sql.{Column, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, StructType}
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

  private def checkClassificationLabels(
      labelCol: String,
      numClasses: Option[Int]): Column = {
    val casted = col(labelCol).cast(DoubleType)
    numClasses match {
      case Some(2) =>
        when(casted.isNull || casted.isNaN, raise_error(lit("Labels MUST NOT be Null or NaN")))
          .when(casted =!= 0 && casted =!= 1,
            raise_error(concat(lit("Labels MUST be in {0, 1}, but got "), casted)))
          .otherwise(casted)

      case _ =>
        val n = numClasses.getOrElse(Int.MaxValue)
        require(0 < n && n <= Int.MaxValue)
        when(casted.isNull || casted.isNaN, raise_error(lit("Labels MUST NOT be Null or NaN")))
          .when(casted < 0 || casted >= n,
            raise_error(concat(lit(s"Labels MUST be in [0, $n), but got "), casted)))
          .when(casted =!= casted.cast(IntegerType),
            raise_error(concat(lit("Labels MUST be Integers, but got "), casted)))
          .otherwise(casted)
    }
  }

  // Copied from DatasetUtils of Spark to compatible with spark below 3.4
  def getNumClasses(dataset: Dataset[_], labelCol: String, maxNumClasses: Int = 100): Int = {
    MetadataUtils.getNumClasses(dataset.schema(labelCol)) match {
      case Some(n: Int) => n
      case None =>
        // Get number of classes from dataset itself.
        val maxLabelRow: Array[Row] = dataset
          .select(max(checkClassificationLabels(labelCol, Some(maxNumClasses))))
          .take(1)
        if (maxLabelRow.isEmpty || maxLabelRow(0).get(0) == null) {
          throw new SparkException("ML algorithm was given empty dataset.")
        }
        val maxDoubleLabel: Double = maxLabelRow.head.getDouble(0)
        require((maxDoubleLabel + 1).isValidInt, s"Classifier found max label value =" +
          s" $maxDoubleLabel but requires integers in range [0, ... ${Int.MaxValue})")
        val numClasses = maxDoubleLabel.toInt + 1
        require(numClasses <= maxNumClasses, s"Classifier inferred $numClasses from label values" +
          s" in column $labelCol, but this exceeded the max numClasses ($maxNumClasses) allowed" +
          s" to be inferred from values.  To avoid this error for labels with > $maxNumClasses" +
          s" classes, specify numClasses explicitly in the metadata; this can be done by applying" +
          s" StringIndexer to the label column.")
        numClasses
    }
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

  def isVectorType(dataType: DataType): Boolean = dataType.isInstanceOf[VectorUDT]
}
