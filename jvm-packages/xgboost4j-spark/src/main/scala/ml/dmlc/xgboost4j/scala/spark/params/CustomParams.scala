/*
 Copyright (c) 2014-2026 by Contributors

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

  // customObj and customEval default to null, so every save and load goes through here. The
  // null case is handled without json4s: its AST classes moved between json4s 3.x (Spark 3.5) and
  // 4.x (Spark 4.x), so a jar built against one cannot call json4s on the other. The output is
  // what json4s writes for null, so saved metadata is unchanged.
  override def jsonEncode(value: T): String = {
    if (value == null) {
      "null"
    } else {
      implicit val format = Serialization.formats(Utils.getTypeHintsFromClass(value))
      compact(render(Extraction.decompose(value)))
    }
  }

  override def jsonDecode(json: String): T = {
    if (json == "null") null.asInstanceOf[T] else jsonDecodeT(json)
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
