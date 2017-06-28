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

import java.io.IOException

import com.esotericsoftware.kryo.io.{Output, Input}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import ml.dmlc.xgboost4j.java.{Booster => JBooster}
import ml.dmlc.xgboost4j.java.XGBoostError
import scala.collection.JavaConverters._
import scala.collection.mutable

class Booster private[xgboost4j](private var booster: JBooster)
  extends Serializable  with KryoSerializable {

  /**
    * Set parameter to the Booster.
    *
    * @param key   param name
    * @param value param value
    */
  @throws(classOf[XGBoostError])
  def setParam(key: String, value: AnyRef): Unit = {
    booster.setParam(key, value)
  }

  /**
   * set parameters
   *
   * @param params parameters key-value map
   */
  @throws(classOf[XGBoostError])
  def setParams(params: Map[String, AnyRef]): Unit = {
    booster.setParams(params.asJava)
  }

  /**
   * Update (one iteration)
   *
   * @param dtrain training data
   * @param iter   current iteration number
   */
  @throws(classOf[XGBoostError])
  def update(dtrain: DMatrix, iter: Int): Unit = {
    booster.update(dtrain.jDMatrix, iter)
  }

  /**
   * update with customize obj func
   *
   * @param dtrain training data
   * @param obj    customized objective class
   */
  @throws(classOf[XGBoostError])
  def update(dtrain: DMatrix, obj: ObjectiveTrait): Unit = {
    booster.update(dtrain.jDMatrix, obj)
  }

  /**
   * update with give grad and hess
   *
   * @param dtrain training data
   * @param grad   first order of gradient
   * @param hess   seconde order of gradient
   */
  @throws(classOf[XGBoostError])
  def boost(dtrain: DMatrix, grad: Array[Float], hess: Array[Float]): Unit = {
    booster.boost(dtrain.jDMatrix, grad, hess)
  }

  /**
   * evaluate with given dmatrixs.
   *
   * @param evalMatrixs dmatrixs for evaluation
   * @param evalNames   name for eval dmatrixs, used for check results
   * @param iter        current eval iteration
   * @return eval information
   */
  @throws(classOf[XGBoostError])
  def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], iter: Int)
    : String = {
    booster.evalSet(evalMatrixs.map(_.jDMatrix), evalNames, iter)
  }

  /**
   * evaluate with given customized Evaluation class
   *
   * @param evalMatrixs evaluation matrix
   * @param evalNames   evaluation names
   * @param eval        custom evaluator
   * @return eval information
   */
  @throws(classOf[XGBoostError])
  def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], eval: EvalTrait)
    : String = {
    booster.evalSet(evalMatrixs.map(_.jDMatrix), evalNames, eval)
  }


  /**
   * Predict with data
   *
   * @param data         dmatrix storing the input
   * @param outPutMargin Whether to output the raw untransformed margin value.
   * @param treeLimit    Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @return predict result
   */
  @throws(classOf[XGBoostError])
  def predict(data: DMatrix, outPutMargin: Boolean = false, treeLimit: Int = 0)
      : Array[Array[Float]] = {
    booster.predict(data.jDMatrix, outPutMargin, treeLimit)
  }

  /**
   * Predict the leaf indices
   *
   * @param data      dmatrix storing the input
   * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @return predict result
   * @throws XGBoostError native error
   */
  @throws(classOf[XGBoostError])
  def predictLeaf(data: DMatrix, treeLimit: Int = 0) : Array[Array[Float]] = {
    booster.predictLeaf(data.jDMatrix, treeLimit)
  }

  /**
    * Output feature contributions toward predictions of given data
    *
    * @param data      dmatrix storing the input
    * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
    * @return The feature contributions and bias.
    * @throws XGBoostError native error
    */
  @throws(classOf[XGBoostError])
  def predictContrib(data: DMatrix, treeLimit: Int = 0) : Array[Array[Float]] = {
    booster.predictContrib(data.jDMatrix, treeLimit)
  }

  /**
   * save model to modelPath
   *
   * @param modelPath model path
   */
  @throws(classOf[XGBoostError])
  def saveModel(modelPath: String): Unit = {
    booster.saveModel(modelPath)
  }
  /**
    * save model to Output stream
    *
    * @param out Output stream
    */
  @throws(classOf[XGBoostError])
  def saveModel(out: java.io.OutputStream): Unit = {
    booster.saveModel(out)
  }
  /**
   * Dump model as Array of string
   *
   * @param featureMap featureMap file
   * @param withStats  bool
   *                   Controls whether the split statistics are output.
   */
  @throws(classOf[XGBoostError])
  def getModelDump(featureMap: String = null, withStats: Boolean = false, format: String = "text")
    : Array[String] = {
    booster.getModelDump(featureMap, withStats, format)
  }

  /**
   * Get importance of each feature
   *
   * @return featureMap  key: feature index, value: feature importance score
   */
  @throws(classOf[XGBoostError])
  def getFeatureScore(featureMap: String = null): mutable.Map[String, Integer] = {
    booster.getFeatureScore(featureMap).asScala
  }

  def toByteArray: Array[Byte] = {
    booster.toByteArray
  }

  /**
    *  Dispose the booster when it is no longer needed
    */
  def dispose: Unit = {
    booster.dispose()
  }

  override def finalize(): Unit = {
    super.finalize()
    dispose
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeObject(output, booster)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    booster = kryo.readObject(input, classOf[JBooster])
  }
}
