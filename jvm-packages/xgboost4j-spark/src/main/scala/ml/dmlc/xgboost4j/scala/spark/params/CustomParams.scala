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

import org.apache.spark.ml.param.{Param, ParamPair, Params}
import org.json4s.{DefaultFormats, Extraction}
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.jackson.Serialization

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import ml.dmlc.xgboost4j.scala.spark.Utils

/**
 * General spark parameter that includes TypeHints for (de)serialization using json4s.
 */
class CustomGeneralParam[T: Manifest](parent: Params,
                                      name: String,
                                      doc: String) extends Param[T](parent, name, doc) {

  /** Creates a param pair with the given value (for Java). */
  override def w(value: T): ParamPair[T] = super.w(value)

  override def jsonEncode(value: T): String = {
    implicit val format = Serialization.formats(Utils.getTypeHintsFromClass(value))
    compact(render(Extraction.decompose(value)))
  }

  override def jsonDecode(json: String): T = {
    jsonDecodeT(json)
  }

  private def jsonDecodeT[T](jsonString: String)(implicit m: Manifest[T]): T = {
    val json = parse(jsonString)
    implicit val formats = DefaultFormats.withHints(Utils.getTypeHintsFromJsonClass(json))
    json.extract[T]
  }
}

class CustomEvalParam(parent: Params,
                      name: String,
                      doc: String) extends CustomGeneralParam[EvalTrait](parent, name, doc)

class CustomObjParam(parent: Params,
                     name: String,
                     doc: String) extends CustomGeneralParam[ObjectiveTrait](parent, name, doc)
