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

package ml.dmlc.xgboost4j.scala.spark.params

import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import org.apache.spark.ml.param.{BooleanParam, Param, Params}

trait GpuParams extends Params {
  /**
   * Param for the names of feature columns.
   * @group param
   */
  final val featuresCols: StringSeqParam = new StringSeqParam(this, "featuresCols",
    "a sequence of feature column names.")

  setDefault(featuresCols, Seq.empty[String])

  /** @group getParam */
  final def getFeaturesCols: Seq[String] = $(featuresCols)

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
