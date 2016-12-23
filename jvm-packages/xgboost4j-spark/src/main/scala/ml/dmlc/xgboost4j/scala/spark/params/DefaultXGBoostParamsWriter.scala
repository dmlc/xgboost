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

import ml.dmlc.xgboost4j.scala.spark.params.Utils.classForName
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.MLReader
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject, _}

// This originates from spark DefaultXGBoostParamsWriter & DefaultParamsReader copy paste
private[xgboost4j] object DefaultXGBoostParamsWriter {

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
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
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

/**
  * Default [[MLReader]] implementation for transformers and estimators that contain basic
  * (json4s-serializable) params and no data. This will not handle more complex params or types with
  * data (e.g., models with coefficients).
  *
  * @tparam T ML instance type
  *           TODO: Consider adding check for correct class name.
  */
private[xgboost4j] class DefaultXGBoostParamsReader[T] extends MLReader[T] {

  override def load(path: String): T = {
    val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
    val cls = classForName(metadata.className)
    val instance =
      cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
    DefaultXGBoostParamsReader.getAndSetParams(instance, metadata)
    instance.asInstanceOf[T]
  }
}

private[xgboost4j] object DefaultXGBoostParamsReader {

  /**
    * All info from metadata file.
    *
    * @param params       paramMap, as a [[JValue]]
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
      * This can be useful for getting a Param value before an instance of [[Params]]
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
          val param = instance.getParam(paramName)
          val value = param.jsonDecode(compact(render(jsonValue)))
          instance.set(param, value)
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
    }
  }

  /**
    * Load a [[Params]] instance from the given path, and return it.
    * This assumes the instance implements [[org.apache.spark.ml.util.MLReadable]].
    */
  def loadParamsInstance[T](path: String, sc: SparkContext): T = {
    val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc)
    val cls = classForName(metadata.className)
    cls.getMethod("read").invoke(null).asInstanceOf[MLReader[T]].load(path)
  }
}
