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

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.ObjectiveTrait
import ml.dmlc.xgboost4j.scala.spark.params.CustomObjParam._

abstract class SparkCustomObj extends ObjectiveTrait {
      SparkCustomObj.addTypeHint(this)
}

object SparkCustomObj {
  private var typeHintAdded = false
  def addTypeHint(thiz: SparkCustomObj): Unit = {
    if (!typeHintAdded) {
      addTypeHintForClass(thiz.getClass())
      typeHintAdded = true
    }
  }
}
