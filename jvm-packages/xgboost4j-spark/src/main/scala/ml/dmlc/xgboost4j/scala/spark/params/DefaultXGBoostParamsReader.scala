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
import org.json4s.{DefaultFormats, JValue}
import org.json4s.JsonAST.JObject
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.util.MLReader

// This originates from apache-spark DefaultPramsReader copy paste
private[spark] object DefaultXGBoostParamsReader {

  private val paramNameCompatibilityMap: Map[String, String] = Map("silent" -> "verbosity")

  private val paramValueCompatibilityMap: Map[String, Map[Any, Any]] =
    Map("objective" -> Map("reg:linear" -> "reg:squarederror"))

  /**
   * All info from metadata file.
   *
   * @param params       paramMap, as a `JValue`
   * @param metadata     All metadata, including the other fields
   * @param metadataJson Full metadata file String (for debugging)
   */
  case class Metadata(
      className: String,
      uid: String,
      timestamp: Long,
      sparkVersion: String,
      params: JValue,
      metadata: JValue,
      metadataJson: String) {

    /**
     * Get the JSON value of the [[org.apache.spark.ml.param.Param]] of the given name.
     * This can be useful for getting a Param value before an instance of `Params`
     * is available.
     */
    def getParamValue(paramName: String): JValue = {
      implicit val format = DefaultFormats
      params match {
        case JObject(pairs) =>
          val values = pairs.filter { case (pName, jsonValue) =>
            pName == paramName
          }.map(_._2)
          assert(values.length == 1, s"Expected one instance of Param '$paramName' but found" +
            s" ${values.length} in JSON Params: " + pairs.map(_.toString).mkString(", "))
          values.head
        case _ =>
          throw new IllegalArgumentException(
            s"Cannot recognize JSON metadata: $metadataJson.")
      }
    }
  }

  /**
   * Load metadata saved using [[DefaultXGBoostParamsWriter.saveMetadata()]]
   *
   * @param expectedClassName If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr, expectedClassName)
  }

  /**
   * Parse metadata JSON string produced by [[DefaultXGBoostParamsWriter.getMetadataToSave()]].
   * This is a helper function for [[loadMetadata()]].
   *
   * @param metadataStr       JSON string of metadata
   * @param expectedClassName If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]
    val params = metadata \ "paramMap"
    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }

    Metadata(className, uid, timestamp, sparkVersion, params, metadata, metadataStr)
  }

  private def handleBrokenlyChangedValue[T](paramName: String, value: T): T = {
    paramValueCompatibilityMap.getOrElse(paramName, Map()).getOrElse(value, value).asInstanceOf[T]
  }

  private def handleBrokenlyChangedName(paramName: String): String = {
    paramNameCompatibilityMap.getOrElse(paramName, paramName)
  }

  /**
   * Extract Params from metadata, and set them in the instance.
   * This works if all Params implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
   * TODO: Move to [[Metadata]] method
   */
  def getAndSetParams(instance: Params, metadata: Metadata): Unit = {
    implicit val format = DefaultFormats
    metadata.params match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          val param = instance.getParam(handleBrokenlyChangedName(paramName))
          val value = param.jsonDecode(compact(render(jsonValue)))
          instance.set(param, handleBrokenlyChangedValue(paramName, value))
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
    }
  }

  /**
   * Load a `Params` instance from the given path, and return it.
   * This assumes the instance implements [[org.apache.spark.ml.util.MLReadable]].
   */
  def loadParamsInstance[T](path: String, sc: SparkContext): T = {
    val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
    val cls = Utils.classForName(metadata.className)
    cls.getMethod("read").invoke(null).asInstanceOf[MLReader[T]].load(path)
  }
}

