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

import java.io.DataInputStream

private[xgboost4j] object DMatrixBuilder extends Serializable {

  def buildDMatrixfromBinaryData(inStream: DataInputStream): DMatrix = {
    // TODO: currently it is random statement for making compiler happy
    new DMatrix(new Array[Float](1), 1, 1)
  }

  def buildDMatrixfromBinaryData(binaryArray: Array[Byte]): DMatrix = {
    // TODO: currently it is random statement for making compiler happy
    new DMatrix(new Array[Float](1), 1, 1)
  }
}
