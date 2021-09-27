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

package ml.dmlc.xgboost4j.scala.spark.rapids

import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, NumericType, StructType}

private[spark] object MLUtils {

  private[rapids] def checkNumericType(schema: StructType, colName: String,
      msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.isInstanceOf[NumericType],
      s"Column $colName must be of type NumericType but was actually of type " +
      s"${actualDataType.catalogString}.$message")
  }

  def prepareColumnType(dataset: Dataset[_],
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

  private[rapids] def validateSchema(schema: StructType,
      featureNames: Seq[String],
      labelName: String = "",
      weightName: String = "",
      marginName: String = "",
      fitting: Boolean = true): StructType = {
    // feature columns
    require(featureNames.nonEmpty, "No feature column name is specified!")
    featureNames.foreach(fn => checkNumericType(schema, fn))
    if (fitting) {
      require(labelName.nonEmpty, "No label column is specified!")
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

  // get column name, null | undefined will be casted to ""
  def getColumnName(params: Params)(param: Param[String]): String = {
    if (params.isDefined(param)) {
      val colName = params.getOrDefault(param)
      if (colName != null) colName else ""
    } else ""
  }

  def getColumnNames(params: Params)(cols: Param[String]*): Seq[String] = {
    val getName = getColumnName(params)(_)
    cols.map(getName)
  }

}
