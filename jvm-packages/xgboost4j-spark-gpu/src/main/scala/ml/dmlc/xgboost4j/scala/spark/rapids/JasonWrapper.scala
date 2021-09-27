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

package ml.dmlc.xgboost4j.scala.spark.rapids

import org.json4s.jackson.JsonMethods
import org.json4s.{JValue, JsonInput}

// The wrapper of some methods of Jason4s for compatibility issue
private[spark] object JasonWrapper {

  def parse(
    in: JsonInput,
    useBigDecimalForDouble: Boolean = false,
    useBigIntForLong: Boolean = true): JValue = {
    val mName = "parse"
    // scalastyle:off classforname
    val clazz = Class.forName("org.json4s.jackson.JsonMethods$")
    // scalastyle:on classforname
    val obj = clazz.getField("MODULE$").get(clazz).asInstanceOf[JsonMethods]
    try {
      // try new version first
      clazz.getDeclaredMethod(mName, classOf[JsonInput], classOf[Boolean], classOf[Boolean])
        .invoke(obj, in, useBigDecimalForDouble.asInstanceOf[AnyRef],
          useBigIntForLong.asInstanceOf[AnyRef])
        .asInstanceOf[JValue]
    } catch {
      case _: NoSuchMethodException =>
        // then try old version with two arguments
        clazz.getDeclaredMethod(mName, classOf[JsonInput], classOf[Boolean])
          .invoke(obj, in, useBigDecimalForDouble.asInstanceOf[AnyRef])
          .asInstanceOf[JValue]
    }
  }
}
