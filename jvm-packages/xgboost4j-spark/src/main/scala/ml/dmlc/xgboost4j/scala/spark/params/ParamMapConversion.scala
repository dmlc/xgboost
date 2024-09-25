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

import scala.collection.mutable

import com.google.common.base.CaseFormat
import org.apache.spark.ml.param._

private[spark] trait ParamMapConversion extends NonXGBoostParams {

  /**
   * Convert XGBoost parameters to Spark Parameters
   *
   * @param xgboostParams XGBoost style parameters
   */
  def xgboost2SparkParams(xgboostParams: Map[String, Any]): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      val lowerCamelName = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, paramName)
      val lowerName = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, paramName)
      val qualifiedNames = mutable.Set(paramName, lowerName, lowerCamelName)
      params.find(p => qualifiedNames.contains(p.name)) foreach {
        case p: DoubleParam =>
          set(p.name, paramValue.toString.toDouble)
        case p: BooleanParam =>
          set(p.name, paramValue.toString.toBoolean)
        case p: IntParam =>
          set(p.name, paramValue.toString.toInt)
        case p: FloatParam =>
          set(p.name, paramValue.toString.toFloat)
        case p: LongParam =>
          set(p.name, paramValue.toString.toLong)
        case p: Param[_] =>
          set(p.name, paramValue)
      }
    }
  }

  /**
   * Convert the user-supplied parameters to the XGBoost parameters.
   *
   * Note that this doesn't contain jvm-specific parameters.
   */
  def getXGBoostParams: Map[String, Any] = {
    val xgboostParams = new mutable.HashMap[String, Any]()

    // Only pass user-supplied parameters to xgboost.
    for (param <- params) {
      if (isSet(param) && !nonXGBoostParams.contains(param.name)) {
        xgboostParams += param.name -> $(param)
      }
    }
    xgboostParams.toMap
  }
}
