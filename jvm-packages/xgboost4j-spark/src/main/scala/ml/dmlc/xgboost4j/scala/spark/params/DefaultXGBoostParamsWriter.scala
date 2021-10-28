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

import org.apache.hadoop.fs.Path

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.json4s.jackson.JsonMethods._
import org.json4s.{JArray, JBool, JDouble, JField, JInt, JNothing, JObject, JString, JValue}

import JsonDSLXGBoost._

// This originates from apache-spark DefaultPramsWriter copy paste
private[spark] object DefaultXGBoostParamsWriter {

  /**
   * Saves metadata + Params to: path + "/metadata"
   *  - class
   *  - timestamp
   *  - sparkVersion
   *  - uid
   *  - paramMap
   *  - (optionally, extra metadata)
   *
   * @param extraMetadata Extra metadata to be saved at same level as uid, paramMap, etc.
   * @param paramMap      If given, this is saved in the "paramMap" field.
   *                      Otherwise, all [[org.apache.spark.ml.param.Param]]s are encoded using
   *                      [[org.apache.spark.ml.param.Param.jsonEncode()]].
   */
  def saveMetadata(
    instance: Params,
    path: String,
    sc: SparkContext,
    extraMetadata: Option[JObject] = None,
    paramMap: Option[JValue] = None): Unit = {

    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, sc, extraMetadata, paramMap)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
   * Helper for [[saveMetadata()]] which extracts the JSON to save.
   * This is useful for ensemble models which need to save metadata for many sub-models.
   *
   * @see [[saveMetadata()]] for details on what this includes.
   */
  def getMetadataToSave(
    instance: Params,
    sc: SparkContext,
    extraMetadata: Option[JObject] = None,
    paramMap: Option[JValue] = None): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.filter{
      case ParamPair(p, _) => p != null
    }.map {
      case ParamPair(p, v) =>
        p.name -> parse(p.jsonEncode(v))
    }.toList))
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }
}

// Fix json4s bin-incompatible issue.
// This originates from org.json4s.JsonDSL of 3.6.6
object JsonDSLXGBoost {

  implicit def seq2jvalue[A](s: Iterable[A])(implicit ev: A => JValue): JArray =
    JArray(s.toList.map(ev))

  implicit def map2jvalue[A](m: Map[String, A])(implicit ev: A => JValue): JObject =
    JObject(m.toList.map { case (k, v) => JField(k, ev(v)) })

  implicit def option2jvalue[A](opt: Option[A])(implicit ev: A => JValue): JValue = opt match {
    case Some(x) => ev(x)
    case None => JNothing
  }

  implicit def short2jvalue(x: Short): JValue = JInt(x)
  implicit def byte2jvalue(x: Byte): JValue = JInt(x)
  implicit def char2jvalue(x: Char): JValue = JInt(x)
  implicit def int2jvalue(x: Int): JValue = JInt(x)
  implicit def long2jvalue(x: Long): JValue = JInt(x)
  implicit def bigint2jvalue(x: BigInt): JValue = JInt(x)
  implicit def double2jvalue(x: Double): JValue = JDouble(x)
  implicit def float2jvalue(x: Float): JValue = JDouble(x.toDouble)
  implicit def bigdecimal2jvalue(x: BigDecimal): JValue = JDouble(x.doubleValue)
  implicit def boolean2jvalue(x: Boolean): JValue = JBool(x)
  implicit def string2jvalue(x: String): JValue = JString(x)

  implicit def symbol2jvalue(x: Symbol): JString = JString(x.name)
  implicit def pair2jvalue[A](t: (String, A))(implicit ev: A => JValue): JObject =
    JObject(List(JField(t._1, ev(t._2))))
  implicit def list2jvalue(l: List[JField]): JObject = JObject(l)
  implicit def jobject2assoc(o: JObject): JsonListAssoc = new JsonListAssoc(o.obj)
  implicit def pair2Assoc[A](t: (String, A))(implicit ev: A => JValue): JsonAssoc[A] =
    new JsonAssoc(t)
}

final class JsonAssoc[A](private val left: (String, A)) extends AnyVal {
  def ~[B](right: (String, B))(implicit ev1: A => JValue, ev2: B => JValue): JObject = {
    val l: JValue = ev1(left._2)
    val r: JValue = ev2(right._2)
    JObject(JField(left._1, l) :: JField(right._1, r) :: Nil)
  }

  def ~(right: JObject)(implicit ev: A => JValue): JObject = {
    val l: JValue = ev(left._2)
    JObject(JField(left._1, l) :: right.obj)
  }
  def ~~[B](right: (String, B))(implicit ev1: A => JValue, ev2: B => JValue): JObject =
    this.~(right)
  def ~~(right: JObject)(implicit ev: A => JValue): JObject = this.~(right)
}

final class JsonListAssoc(private val left: List[JField]) extends AnyVal {
  def ~(right: (String, JValue)): JObject = JObject(left ::: List(JField(right._1, right._2)))
  def ~(right: JObject): JObject = JObject(left ::: right.obj)
  def ~~(right: (String, JValue)): JObject = this.~(right)
  def ~~(right: JObject): JObject = this.~(right)
}
