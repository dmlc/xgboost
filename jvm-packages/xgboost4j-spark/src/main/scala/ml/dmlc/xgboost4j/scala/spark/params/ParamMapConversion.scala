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

import org.apache.spark.ml.param._

private[spark] trait ParamMapConversion extends NonXGBoostParams {

  /**
   * Convert XGBoost parameters to Spark Parameters
   *
   * @param xgboostParams XGBoost style parameters
   */
  def xgboost2SparkParams(xgboostParams: Map[String, Any]): Unit = {
    for ((name, paramValue) <- xgboostParams) {
      params.find(_.name == name).foreach {
        case _: DoubleParam =>
          set(name, paramValue.toString.toDouble)
        case _: BooleanParam =>
          set(name, paramValue.toString.toBoolean)
        case _: IntParam =>
          set(name, paramValue.toString.toInt)
        case _: FloatParam =>
          set(name, paramValue.toString.toFloat)
        case _: LongParam =>
          set(name, paramValue.toString.toLong)
        case _: Param[_] =>
          set(name, paramValue)
      }
    }
  }

  /**
   * Convert the user-supplied parameters to the XGBoost parameters.
   *
   * Note that this also contains jvm-specific parameters.
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
