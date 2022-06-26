/*
 Copyright (c) 2014-2022 by Contributors

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

package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{Param, ParamValidators}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasHandleInvalid, HasLabelCol, HasWeightCol}
import org.apache.spark.ml.util.XGBoostSchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

private[scala] sealed trait XGBoostEstimatorCommon extends GeneralParams with LearningTaskParams
  with BoosterParams with RabitParams with ParamMapFuncs with NonParamVariables with HasWeightCol
  with HasBaseMarginCol with HasLeafPredictionCol with HasContribPredictionCol with HasFeaturesCol
  with HasLabelCol with HasFeaturesCols with HasHandleInvalid {

  def needDeterministicRepartitioning: Boolean = {
    isDefined(checkpointPath) && getCheckpointPath != null && getCheckpointPath.nonEmpty &&
      isDefined(checkpointInterval) && getCheckpointInterval > 0
  }

  /**
   * Param for how to handle invalid data (NULL values). Options are 'skip' (filter out rows with
   * invalid data), 'error' (throw an error), or 'keep' (return relevant number of NaN in the
   * output). Column lengths are taken from the size of ML Attribute Group, which can be set using
   * `VectorSizeHint` in a pipeline before `VectorAssembler`. Column lengths can also be inferred
   * from first rows of the data since it is safe to do so but only in case of 'error' or 'skip'.
   * Default: "error"
   * @group param
   */
  override val handleInvalid: Param[String] = new Param[String](this, "handleInvalid",
    """Param for how to handle invalid data (NULL and NaN values). Options are 'skip' (filter out
      |rows with invalid data), 'error' (throw an error), or 'keep' (return relevant number of NaN
      |in the output). Column lengths are taken from the size of ML Attribute Group, which can be
      |set using `VectorSizeHint` in a pipeline before `VectorAssembler`. Column lengths can also
      |be inferred from first rows of the data since it is safe to do so but only in case of 'error'
      |or 'skip'.""".stripMargin.replaceAll("\n", " "),
    ParamValidators.inArray(Array("skip", "error", "keep")))

  setDefault(handleInvalid, "error")

  /**
   * Specify an array of feature column names which must be numeric types.
   */
  def setFeaturesCol(value: Array[String]): this.type = set(featuresCols, value)

  /** Set the handleInvalid for VectorAssembler */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /**
   * Check if schema has a field named with the value of "featuresCol" param and it's data type
   * must be VectorUDT
   */
  def isFeaturesColSet(schema: StructType): Boolean = {
    schema.fieldNames.contains(getFeaturesCol) &&
      XGBoostSchemaUtils.isVectorUDFType(schema(getFeaturesCol).dataType)
  }

  /** check the features columns type */
  def transformSchemaWithFeaturesCols(fit: Boolean, schema: StructType): StructType = {
    if (isFeaturesColsValid) {
      if (fit) {
        XGBoostSchemaUtils.checkNumericType(schema, $(labelCol))
      }
      $(featuresCols).foreach(feature =>
        XGBoostSchemaUtils.checkFeatureColumnType(schema(feature).dataType))
      schema
    } else {
      throw new IllegalArgumentException("featuresCol or featuresCols must be specified")
    }
  }

  /**
   * Vectorize the features columns if necessary.
   *
   * @param input the input dataset
   * @return (output dataset and the feature column name)
   */
  def vectorize(input: Dataset[_]): (Dataset[_], String) = {
    val schema = input.schema
    if (isFeaturesColSet(schema)) {
      // Dataset already has vectorized.
      (input, getFeaturesCol)
    } else if (isFeaturesColsValid) {
      val featuresName = if (!schema.fieldNames.contains(getFeaturesCol)) {
        getFeaturesCol
      } else {
        "features_" + uid
      }
      val vectorAssembler = new VectorAssembler()
        .setHandleInvalid($(handleInvalid))
        .setInputCols(getFeaturesCols)
        .setOutputCol(featuresName)
      (vectorAssembler.transform(input).select(featuresName, getLabelCol), featuresName)
    } else {
      // never reach here, since transformSchema will take care of the case
      // that featuresCols is invalid
      (input, getFeaturesCol)
    }
  }
}

private[scala] trait XGBoostClassifierParams extends XGBoostEstimatorCommon with HasNumClass

private[scala] trait XGBoostRegressorParams extends XGBoostEstimatorCommon with HasGroupCol
