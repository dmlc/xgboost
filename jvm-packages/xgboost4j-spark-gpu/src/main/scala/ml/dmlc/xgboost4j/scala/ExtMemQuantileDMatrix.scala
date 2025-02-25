/*
 Copyright (c) 2025 by Contributors

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

import ml.dmlc.xgboost4j.java.{ColumnBatch, ExtMemQuantileDMatrix => jExtMemQuantileDMatrix}

class ExtMemQuantileDMatrix private[scala](
  private[scala] override val jDMatrix: jExtMemQuantileDMatrix) extends QuantileDMatrix(jDMatrix) {

  def this(iter: Iterator[ColumnBatch],
           missing: Float,
           maxBin: Int,
           ref: Option[QuantileDMatrix],
           nthread: Int,
           maxNumDevicePages: Int,
           maxQuantileBatches: Int,
           minCachePageBytes: Int) {
    this(new jExtMemQuantileDMatrix(iter.asJava, missing, maxBin,
      ref.map(_.jDMatrix).orNull,
      nthread, maxNumDevicePages, maxQuantileBatches, minCachePageBytes))
  }

  def this(iter: Iterator[ColumnBatch], missing: Float, maxBin: Int) {
    this(new jExtMemQuantileDMatrix(iter.asJava, missing, maxBin))
  }

  def this(
    iter: Iterator[ColumnBatch],
    ref: ExtMemQuantileDMatrix,
    missing: Float,
    maxBin: Int
  ) {
    this(new jExtMemQuantileDMatrix(iter.asJava, missing, maxBin, ref.jDMatrix))
  }
}
