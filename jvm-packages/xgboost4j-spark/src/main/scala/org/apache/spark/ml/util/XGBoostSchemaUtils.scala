/*
 Copyright (c) 2022-2023 by Contributors

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

package org.apache.spark.ml.util

import org.apache.spark.sql.types.{BooleanType, DataType, NumericType, StructType}
import org.apache.spark.ml.linalg.VectorUDT

object XGBoostSchemaUtils {

  /** check if the dataType is VectorUDT */
  def isVectorUDFType(dataType: DataType): Boolean = {
    dataType match {
      case _: VectorUDT => true
      case _ => false
    }
  }

  /** The feature columns will be vectorized by VectorAssembler first, which only
   * supports Numeric, Boolean and VectorUDT types */
  def checkFeatureColumnType(dataType: DataType): Unit = {
    dataType match {
      case _: NumericType | BooleanType =>
      case _: VectorUDT =>
      case d => throw new UnsupportedOperationException(s"featuresCols only supports Numeric, " +
        s"boolean and VectorUDT types, found: ${d}")
    }
  }

  def checkNumericType(
      schema: StructType,
      colName: String,
      msg: String = ""): Unit = {
    SchemaUtils.checkNumericType(schema, colName, msg)
  }

}
