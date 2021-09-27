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

import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import org.apache.spark.ml.param.{BooleanParam, Param, Params}

trait GpuParams extends Params {
  /**
   * Param for the names of multiple feature columns.
   * @group param
   */
  final val featuresCols: StringSeqParam = new StringSeqParam(this, "featuresCols",
    "names of multiple feature columns.")

  setDefault(featuresCols, Seq.empty[String])

  /** @group getParam */
  final def getFeaturesCols: Seq[String] = $(featuresCols)

  /**
   * Param for the names of columns to be converted to row.
   * If not specified, all the supported columns will be included.
   * @group param
   */
  final val toRowCols: StringSeqParam = new StringSeqParam(this, "toRowCols",
    "names of columns to be converted to row")

  setDefault(toRowCols, Seq.empty[String])

  final def getToRowCols: Seq[String] = $(toRowCols)

  final val buildAllColumnsInTransform = new BooleanParam(this,
    "buildAllColumnsInTransform", "whether building all columns when transform." +
      " when set to false, only columns with numeric type can be built. Defaut to true")

  // default to true
  setDefault(buildAllColumnsInTransform, true)

  final def getBuildAllColumnsInTransform: Boolean = $(buildAllColumnsInTransform)
}

class StringSeqParam(
  parent: Params,
  name: String,
  doc: String) extends Param[Seq[String]](parent, name, doc) {

  override def jsonEncode(value: Seq[String]): String = {
    import org.json4s.JsonDSL._
    compact(render(value))
  }

  override def jsonDecode(json: String): Seq[String] = {
    implicit val formats = DefaultFormats
    parse(json).extract[Seq[String]]
  }
}
