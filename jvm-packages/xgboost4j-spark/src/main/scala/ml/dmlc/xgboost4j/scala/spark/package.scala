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

package ml.dmlc.xgboost4j.scala

import java.util.Properties

import org.apache.spark.SparkException

package object spark {
  private def loadVersionInfo(): String = {
    val versionResourceFile = Thread.currentThread().getContextClassLoader.getResourceAsStream(
      "xgboost4j-version.properties")
    try {
      val unknownProp = "<unknown>"
      val props = new Properties()
      props.load(versionResourceFile)
      props.getProperty("version", unknownProp)
    } catch {
      case e: Exception =>
        throw new SparkException("Error loading properties from xgboost4j-version.properties", e)
    } finally {
      if (versionResourceFile != null) {
        try {
          versionResourceFile.close()
        } catch {
          case e: Exception =>
            throw new SparkException("Error closing xgboost4j version resource stream", e)
        }
      }
    }
  }

  val VERSION: String = loadVersionInfo()
}
