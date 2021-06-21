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

/**
 * Tensor of Scala version
 * @param tensor Tensor of Java version
 */
case class Tensor(tensor: ml.dmlc.xgboost4j.java.Tensor) {

  /** Get the dimension of the Tensor */
  def getDimension: Int = dim

  /** Get the predict result */
  def getPredictResult: Array[_] = toArray

  private lazy val dim = tensor.getDim.toInt

  private lazy val toArray: Array[_] = {
    val result = tensor.getRawResult
    val shape = tensor.getShape
    val dim = tensor.getDim

    if (dim == 1) {
      result.grouped(1).toArray
    } else if (dim == 2) {
      result.grouped(shape(1).toInt).toArray
    } else if (dim == 3) {
      val array = Array.ofDim[Float](shape(0).toInt, shape(1).toInt, shape(2).toInt)
      for (i <- 0 until shape(0).toInt) {
        for (j <- 0 until shape(1).toInt) {
          for (k <- 0 until shape(2).toInt) {
            val index = i * shape(1).toInt + j * shape(2) + k
            array(i)(j)(k) = result(index.toInt)
          }
        }
      }
      array
    } else if (dim == 4) {
      val array = Array.ofDim[Float](shape(0).toInt, shape(1).toInt, shape(2).toInt, shape(3).toInt)
      for (i <- 0 until shape(0).toInt) {
        for (j <- 0 until shape(1).toInt) {
          for (k <- 0 until shape(2).toInt) {
            for (m <- 0 until shape(3).toInt) {
              val index = i * shape(1) + j * shape(2) + k * shape(3) + m
              array(i)(j)(k)(m) = result(index.toInt)
            }
          }
        }
      }
      array
    } else {
      throw new IllegalArgumentException("Wrong dimension")
    }
  }
}
