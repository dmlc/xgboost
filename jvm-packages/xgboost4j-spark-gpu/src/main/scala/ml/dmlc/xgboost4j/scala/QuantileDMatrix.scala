/*
 Copyright (c) 2021-2024 by Contributors

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

import ml.dmlc.xgboost4j.java.{Column, ColumnBatch, QuantileDMatrix => JQuantileDMatrix, XGBoostError}

class QuantileDMatrix private[scala](
    private[scala] override val jDMatrix: JQuantileDMatrix) extends DMatrix(jDMatrix) {

  /**
   * Create QuantileDMatrix from iterator based on the array interface
   *
   * @param iter    the XGBoost ColumnBatch batch to provide the corresponding array interface
   * @param missing the missing value
   * @param maxBin  the max bin
   * @param nthread the parallelism
   * @throws XGBoostError
   */
  def this(iter: Iterator[ColumnBatch], missing: Float, maxBin: Int, nthread: Int) {
    this(new JQuantileDMatrix(iter.asJava, missing, maxBin, nthread))
  }

  /**
   * Create QuantileDMatrix from iterator based on the array interface
   *
   * @param iter       the XGBoost ColumnBatch batch to provide the corresponding array interface
   * @param refDMatrix The reference QuantileDMatrix that provides quantile information, needed
   *                   when creating validation/test dataset with QuantileDMatrix. Supplying the
   *                   training DMatrix as a reference means that the same quantisation applied
   *                   to the training data is applied to the validation/test data
   * @param missing    the missing value
   * @param maxBin     the max bin
   * @param nthread    the parallelism
   * @throws XGBoostError
   */
  def this(iter: Iterator[ColumnBatch],
           ref: QuantileDMatrix,
           missing: Float,
           maxBin: Int,
           nthread: Int) {
    this(new JQuantileDMatrix(iter.asJava, ref.jDMatrix, missing, maxBin, nthread))
  }

  /**
   * set label of dmatrix
   *
   * @param labels labels
   */
  @throws(classOf[XGBoostError])
  override def setLabel(labels: Array[Float]): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setLabel.")

  /**
   * set weight of each instance
   *
   * @param weights weights
   */
  @throws(classOf[XGBoostError])
  override def setWeight(weights: Array[Float]): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setWeight.")

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   */
  @throws(classOf[XGBoostError])
  override def setBaseMargin(baseMargin: Array[Float]): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.")

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   */
  @throws(classOf[XGBoostError])
  override def setBaseMargin(baseMargin: Array[Array[Float]]): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.")

  /**
   * Set group sizes of DMatrix (used for ranking)
   *
   * @param group group size as array
   */
  @throws(classOf[XGBoostError])
  override def setGroup(group: Array[Int]): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setGroup.")

  /**
   * Set label of DMatrix from array interface
   */
  @throws(classOf[XGBoostError])
  override def setLabel(column: Column): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setLabel.")

  /**
   * set weight of dmatrix from column array interface
   */
  @throws(classOf[XGBoostError])
  override def setWeight(column: Column): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setWeight.")

  /**
   * set base margin of dmatrix from column array interface
   */
  @throws(classOf[XGBoostError])
  override def setBaseMargin(column: Column): Unit =
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.")

  @throws(classOf[XGBoostError])
  override def setQueryId(column: Column): Unit = {
    throw new XGBoostError("QuantileDMatrix does not support setQueryId.")
  }

}
