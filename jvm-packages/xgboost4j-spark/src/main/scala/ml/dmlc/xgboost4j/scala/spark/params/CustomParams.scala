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

package ml.dmlc.xgboost4j.scala.spark.params

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import ml.dmlc.xgboost4j.scala.spark.TrackerConf
import org.json4s.{DefaultFormats, Extraction, NoTypeHints}
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import org.apache.spark.ml.param.{Param, ParamPair, Params}

class GroupDataParam(
    parent: Params,
    name: String,
    doc: String) extends Param[Seq[Seq[Int]]](parent, name, doc) {

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Seq[Seq[Int]]): ParamPair[Seq[Seq[Int]]] = super.w(value)

  override def jsonEncode(value: Seq[Seq[Int]]): String = {
    import org.json4s.jackson.Serialization
    implicit val formats = Serialization.formats(NoTypeHints)
    compact(render(Extraction.decompose(value)))
  }

  override def jsonDecode(json: String): Seq[Seq[Int]] = {
    implicit val formats = DefaultFormats
    parse(json).extract[Seq[Seq[Int]]]
  }
}

class CustomEvalParam(
    parent: Params,
    name: String,
    doc: String) extends Param[EvalTrait](parent, name, doc) {

  /** Creates a param pair with the given value (for Java). */
  override def w(value: EvalTrait): ParamPair[EvalTrait] = super.w(value)

  override def jsonEncode(value: EvalTrait): String = {
    import org.json4s.jackson.Serialization
    implicit val formats = Serialization.formats(NoTypeHints)
    compact(render(Extraction.decompose(value)))
  }

  override def jsonDecode(json: String): EvalTrait = {
    implicit val formats = DefaultFormats
    parse(json).extract[EvalTrait]
  }
}

class CustomObjParam(
    parent: Params,
    name: String,
    doc: String) extends Param[ObjectiveTrait](parent, name, doc) {

  /** Creates a param pair with the given value (for Java). */
  override def w(value: ObjectiveTrait): ParamPair[ObjectiveTrait] = super.w(value)

  override def jsonEncode(value: ObjectiveTrait): String = {
    import org.json4s.jackson.Serialization
    implicit val formats = Serialization.formats(NoTypeHints)
    compact(render(Extraction.decompose(value)))
  }

  override def jsonDecode(json: String): ObjectiveTrait = {
    implicit val formats = DefaultFormats
    parse(json).extract[ObjectiveTrait]
  }
}

class TrackerConfParam(
    parent: Params,
    name: String,
    doc: String) extends Param[TrackerConf](parent, name, doc) {

  /** Creates a param pair with the given value (for Java). */
  override def w(value: TrackerConf): ParamPair[TrackerConf] = super.w(value)

  override def jsonEncode(value: TrackerConf): String = {
    import org.json4s.jackson.Serialization
    implicit val formats = Serialization.formats(NoTypeHints)
    compact(render(Extraction.decompose(value)))
  }

  override def jsonDecode(json: String): TrackerConf = {
    implicit val formats = DefaultFormats
    val parsedValue = parse(json)
    println(parsedValue.children)
    parsedValue.extract[TrackerConf]
  }
}
