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

package ml.dmlc.xgboost4j.scala.spark.util

import org.json4s.{DefaultFormats, FullTypeHints, JField, JValue, NoTypeHints, TypeHints}

// based on org.apache.spark.util copy /paste
object Utils {

  def getSparkClassLoader: ClassLoader = getClass.getClassLoader

  def getContextOrSparkClassLoader: ClassLoader =
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getSparkClassLoader)

  // scalastyle:off classforname
  /** Preferred alternative to Class.forName(className) */
  def classForName(className: String): Class[_] = {
    Class.forName(className, true, getContextOrSparkClassLoader)
    // scalastyle:on classforname
  }

  /**
   * Get the TypeHints according to the value
   * @param value the instance of class to be serialized
   * @return if value is null,
   *            return NoTypeHints
   *         else return the FullTypeHints.
   *
   *         The FullTypeHints will save the full class name into the "jsonClass" of the json,
   *         so we can find the jsonClass and turn it to FullTypeHints when deserializing.
   */
  def getTypeHintsFromClass(value: Any): TypeHints = {
    if (value == null) { // XGBoost will save the default value (null)
      NoTypeHints
    } else {
      FullTypeHints(List(value.getClass))
    }
  }

  /**
   * Get the TypeHints according to the saved jsonClass field
   * @param json
   * @return TypeHints
   */
  def getTypeHintsFromJsonClass(json: JValue): TypeHints = {
    val jsonClassField = json findField {
      case JField("jsonClass", _) => true
      case _ => false
    }

    jsonClassField.map { field =>
      implicit val formats = DefaultFormats
      val className = field._2.extract[String]
      FullTypeHints(List(Utils.classForName(className)))
    }.getOrElse(NoTypeHints)
  }
}
