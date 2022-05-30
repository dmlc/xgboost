/*
 Copyright (c) 2022 by Contributors

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

package org.apache.spark.ml.util

import ml.dmlc.xgboost4j.java.{Booster => JBooster}
import ml.dmlc.xgboost4j.scala.spark
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.FSDataInputStream
import org.json4s.DefaultFormats
import org.json4s.JsonAST.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, render}

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.DefaultParamsReader.Metadata

abstract class XGBoostWriter extends MLWriter {

  /** Currently it's using the "deprecated" format as
   * default, which will be changed into `ubj` in future releases. */
  def getModelFormat(): String = {
    optionMap.getOrElse("format", JBooster.DEFAULT_FORMAT)
  }
}

object DefaultXGBoostParamsWriter {

  val XGBOOST_VERSION_TAG = "xgboostVersion"

  /**
   * Saves metadata + Params to: path + "/metadata" using [[DefaultParamsWriter.saveMetadata]]
   */
  def saveMetadata(
    instance: Params,
    path: String,
    sc: SparkContext): Unit = {
    // save xgboost version to distinguish the old model.
    val extraMetadata: JObject = Map(XGBOOST_VERSION_TAG -> ml.dmlc.xgboost4j.scala.spark.VERSION)
    DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
  }
}

object DefaultXGBoostParamsReader {

  private val logger = LogFactory.getLog("XGBoostSpark")

  /**
   * Load metadata saved using [[DefaultParamsReader.loadMetadata()]]
   *
   * @param expectedClassName If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
  }

  /**
   * Extract Params from metadata, and set them in the instance.
   * This works if all Params implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
   *
   * And it will auto-skip the parameter not defined.
   *
   * This API is mainly copied from DefaultParamsReader
   */
  def getAndSetParams(instance: Params, metadata: Metadata): Unit = {

    // XGBoost didn't set the default parameters since the save/load code is copied
    // from spark 2.3.x, which means it just used the default values
    // as the same with XGBoost version instead of them in model.
    // For the compatibility, here we still don't set the default parameters.
    //    setParams(instance, metadata, isDefault = true)

    setParams(instance, metadata, isDefault = false)
  }

  /** This API is only for XGBoostClassificationModel */
  def getNumClass(metadata: Metadata, dataInStream: FSDataInputStream): Int = {
    implicit val format = DefaultFormats

    // The xgboostVersion in the meta can specify if the model is the old xgboost in-compatible
    // or the new xgboost compatible.
    val xgbVerOpt = (metadata.metadata \ DefaultXGBoostParamsWriter.XGBOOST_VERSION_TAG)
      .extractOpt[String]

    // For binary:logistic, the numClass parameter can't be set to 2 or not be set.
    // For multi:softprob or multi:softmax, the numClass parameter must be set correctly,
    //   or else, XGBoost will throw exception.
    // So it's safe to get numClass from meta data.
    xgbVerOpt
      .map { _ => (metadata.params \ "numClass").extractOpt[Int].getOrElse(2) }
      .getOrElse(dataInStream.readInt())

  }

  private def setParams(
      instance: Params,
      metadata: Metadata,
      isDefault: Boolean): Unit = {
    val paramsToSet = if (isDefault) metadata.defaultParams else metadata.params
    paramsToSet match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          val finalName = handleBrokenlyChangedName(paramName)
          // For the deleted parameters, we'd better to remove it instead of throwing an exception.
          // So we need to check if the parameter exists instead of blindly setting it.
          if (instance.hasParam(finalName)) {
            val param = instance.getParam(finalName)
            val value = param.jsonDecode(compact(render(jsonValue)))
            instance.set(param, handleBrokenlyChangedValue(paramName, value))
          } else {
            logger.warn(s"$finalName is no longer used in ${spark.VERSION}")
          }
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
    }
  }

  private val paramNameCompatibilityMap: Map[String, String] = Map("silent" -> "verbosity")

  /** This is really not good to do this transformation, but it is needed since there're
   * some tests based on 0.82 saved model in which the objective is "reg:linear" */
  private val paramValueCompatibilityMap: Map[String, Map[Any, Any]] =
    Map("objective" -> Map("reg:linear" -> "reg:squarederror"))

  private def handleBrokenlyChangedName(paramName: String): String = {
    paramNameCompatibilityMap.getOrElse(paramName, paramName)
  }

  private def handleBrokenlyChangedValue[T](paramName: String, value: T): T = {
    paramValueCompatibilityMap.getOrElse(paramName, Map()).getOrElse(value, value).asInstanceOf[T]
  }

}
