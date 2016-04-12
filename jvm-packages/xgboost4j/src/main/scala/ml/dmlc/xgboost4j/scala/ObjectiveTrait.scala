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

import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.java.IObjective

trait ObjectiveTrait extends IObjective {
  /**
   * user define objective function, return gradient and second order gradient
   *
   * @param predicts untransformed margin predicts
   * @param dtrain   training data
   * @return List with two float array, correspond to first order grad and second order grad
   */
  def getGradient(predicts: Array[Array[Float]], dtrain: DMatrix): List[Array[Float]]

  private[scala] def getGradient(predicts: Array[Array[Float]], dtrain: JDMatrix):
    java.util.List[Array[Float]] = {
    getGradient(predicts, new DMatrix(dtrain)).asJava
  }
}
