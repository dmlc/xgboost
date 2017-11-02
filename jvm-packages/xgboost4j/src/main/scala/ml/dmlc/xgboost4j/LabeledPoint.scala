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

package ml.dmlc.xgboost4j

/**
 * Labeled training data point.
 *
 * @param label Label of this point.
 * @param indices Feature indices of this point or `null` if the data is dense.
 * @param values Feature values of this point.
 * @param weight Weight of this point.
 * @param group Group of this point (used for ranking) or -1.
 * @param baseMargin Initial prediction on this point or `Float.NaN`
 */
case class LabeledPoint(
    label: Float,
    indices: Array[Int],
    values: Array[Float],
    weight: Float = 1f,
    group: Int = -1,
    baseMargin: Float = Float.NaN) extends Serializable {
  require(indices == null || indices.length == values.length,
    "indices and values must have the same number of elements")

  def this(label: Float, indices: Array[Int], values: Array[Float]) = {
    // [[weight]] default duplicated to disambiguate the constructor call.
    this(label, indices, values, 1.0f)
  }
}
