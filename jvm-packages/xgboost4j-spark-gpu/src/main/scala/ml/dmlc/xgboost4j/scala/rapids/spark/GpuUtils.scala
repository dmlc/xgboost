/*
 Copyright (c) 2021 by Contributors

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

package ml.dmlc.xgboost4j.scala.rapids.spark

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.{ColumnarRdd, GpuColumnVectorUtils}
import ml.dmlc.xgboost4j.scala.spark.util.Utils

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataType, FloatType, NumericType, StructType}
import org.apache.spark.sql.vectorized.ColumnVector

private[spark] object GpuUtils {

  def extractBatchToHost(table: Table, types: Array[DataType]): Array[ColumnVector] = {
    // spark-rapids has shimmed the GpuColumnVector from 22.10
    GpuColumnVectorUtils.extractHostColumns(table, types)
  }

  def toColumnarRdd(df: DataFrame): RDD[Table] = ColumnarRdd(df)

  def seqIntToSeqInteger(x: Seq[Int]): Seq[Integer] = x.map(new Integer(_))

  /** APIs for gpu column data related */
  def buildColumnDataBatch(featureNames: Seq[String],
      labelName: String,
      weightName: String,
      marginName: String,
      groupName: String,
      dataFrame: DataFrame): ColumnDataBatch = {
    // Some check first
    val schema = dataFrame.schema
    val featureNameSet = featureNames.distinct
    GpuUtils.validateSchema(schema, featureNameSet, labelName, weightName, marginName)

    // group column
    val (opGroup, groupId) = if (groupName.isEmpty) {
      (None, None)
    } else {
      GpuUtils.checkNumericType(schema, groupName)
      (Some(groupName), Some(schema.fieldIndex(groupName)))
    }
    // weight and base margin columns
    val Seq(weightId, marginId) = Seq(weightName, marginName).map {
      name =>
        if (name.isEmpty) None else Some(schema.fieldIndex(name))
    }

    val colsIndices = ColumnIndices(featureNameSet.map(schema.fieldIndex),
      schema.fieldIndex(labelName), weightId, marginId, groupId)
    ColumnDataBatch(dataFrame, colsIndices, opGroup)
  }

  def checkNumericType(schema: StructType, colName: String,
      msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.isInstanceOf[NumericType],
      s"Column $colName must be of NumericType but found: " +
        s"${actualDataType.catalogString}.$message")
  }

  /** Check and Cast the columns to FloatType */
  def prepareColumnType(
      dataset: Dataset[_],
      featureNames: Seq[String],
      labelName: String = "",
      weightName: String = "",
      marginName: String = "",
      fitting: Boolean = true): DataFrame = {
    // check first
    val featureNameSet = featureNames.distinct
    validateSchema(dataset.schema, featureNameSet, labelName, weightName, marginName, fitting)

    val castToFloat = (ds: Dataset[_], colName: String) => {
      val colMeta = ds.schema(colName).metadata
      ds.withColumn(colName, col(colName).as(colName, colMeta).cast(FloatType))
    }
    val colNames = if (fitting) {
      var names = featureNameSet :+ labelName
      if (weightName.nonEmpty) {
        names = names :+ weightName
      }
      if (marginName.nonEmpty) {
        names = names :+ marginName
      }
      names
    } else {
      featureNameSet
    }
    colNames.foldLeft(dataset.asInstanceOf[DataFrame])(
      (ds, colName) => castToFloat(ds, colName))
  }

  /** Validate input schema  */
  def validateSchema(schema: StructType,
      featureNames: Seq[String],
      labelName: String = "",
      weightName: String = "",
      marginName: String = "",
      fitting: Boolean = true): StructType = {
    val msg = if (fitting) "train" else "transform"
    // feature columns
    require(featureNames.nonEmpty, s"Gpu $msg requires features columns. " +
      "please refer to `setFeaturesCol(value: Array[String])`!")
    featureNames.foreach(fn => checkNumericType(schema, fn))
    if (fitting) {
      require(labelName.nonEmpty, "label column is not set.")
      checkNumericType(schema, labelName)

      if (weightName.nonEmpty) {
        checkNumericType(schema, weightName)
      }
      if (marginName.nonEmpty) {
        checkNumericType(schema, marginName)
      }
    }
    schema
  }

  def time[R](block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    (result, (t1 - t0).toFloat / 1000)
  }

  /** Get column names from Parameter */
  def getColumnNames(params: Params)(cols: Param[String]*): Seq[String] = {
    // get column name, null | undefined will be casted to ""
    def getColumnName(params: Params)(param: Param[String]): String = {
      if (params.isDefined(param)) {
        val colName = params.getOrDefault(param)
        if (colName != null) colName else ""
      } else ""
    }

    val getName = getColumnName(params)(_)
    cols.map(getName)
  }

}

/**
 * A container to contain the column ids
 */
private[spark] case class ColumnIndices(
  featureIds: Seq[Int],
  labelId: Int,
  weightId: Option[Int],
  marginId: Option[Int],
  groupId: Option[Int])

private[spark] case class ColumnDataBatch(
  rawDF: DataFrame,
  colIndices: ColumnIndices,
  groupColName: Option[String])
