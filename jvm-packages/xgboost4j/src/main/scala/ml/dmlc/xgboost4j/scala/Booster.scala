/*
 Copyright (c) 2014-2022 by Contributors

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

import com.esotericsoftware.kryo.io.{Output, Input}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import ml.dmlc.xgboost4j.java.{Booster => JBooster}
import ml.dmlc.xgboost4j.java.XGBoostError
import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Booster for xgboost, this is a model API that support interactive build of a XGBoost Model
  *
  * DEVELOPER WARNING: A Java Booster must not be shared by more than one Scala Booster
  * @param booster the java booster object.
  */
class Booster private[xgboost4j](private[xgboost4j] var booster: JBooster)
  extends Serializable  with KryoSerializable {

  /**
   * Get attributes stored in the Booster as a Map.
   *
   * @return A map contain attribute pairs.
   */
  @throws(classOf[XGBoostError])
  def getAttrs: Map[String, String] = {
    booster.getAttrs.asScala.toMap
  }

  /**
   * Get attribute from the Booster.
   *
   * @param key   attr name
   * @return attr value
   */
  @throws(classOf[XGBoostError])
  def getAttr(key: String): String = {
    booster.getAttr(key)
  }

  /**
   * Set attribute to the Booster.
   *
   * @param key   attr name
   * @param value attr value
   */
  @throws(classOf[XGBoostError])
  def setAttr(key: String, value: String): Unit = {
    booster.setAttr(key, value)
  }

  /**
   * set attributes
   *
   * @param params attributes key-value map
   */
  @throws(classOf[XGBoostError])
  def setAttrs(params: Map[String, String]): Unit = {
    booster.setAttrs(params.asJava)
  }

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
  def predict(data: DMatrix, outPutMargin: Boolean = false, treeLimit: Int = 0):
      Array[Array[Float]] = {
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
  def predictLeaf(data: DMatrix, treeLimit: Int = 0): Array[Array[Float]] = {
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
   * save model to Output stream
   * @param out output stream
   * @param format the supported model format, (json, ubj, deprecated)
   * @throws ml.dmlc.xgboost4j.java.XGBoostError
   */
  @throws(classOf[XGBoostError])
  def saveModel(out: java.io.OutputStream, format: String): Unit = {
    booster.saveModel(out, format)
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
    * Dump model as Array of string with specified feature names.
    *
    * @param featureNames Names of features.
    */
  @throws(classOf[XGBoostError])
  def getModelDump(featureNames: Array[String]): Array[String] = {
    booster.getModelDump(featureNames, false, "text")
  }

  def getModelDump(featureNames: Array[String], withStats: Boolean, format: String)
    : Array[String] = {
    booster.getModelDump(featureNames, withStats, format)
  }


  /**
   * Get importance of each feature based on weight only (number of splits)
   *
   * @return featureScoreMap  key: feature index, value: feature importance score
   */
  @throws(classOf[XGBoostError])
  def getFeatureScore(featureMap: String = null): mutable.Map[String, Integer] = {
    booster.getFeatureScore(featureMap).asScala
  }

  /**
    * Get importance of each feature based on weight only
    * (number of splits), with specified feature names.
    *
    * @return featureScoreMap  key: feature name, value: feature importance score
    */
  @throws(classOf[XGBoostError])
  def getFeatureScore(featureNames: Array[String]): mutable.Map[String, Integer] = {
    booster.getFeatureScore(featureNames).asScala
  }

  /**
    * Get importance of each feature based on information gain or cover
    * Supported: ["gain, "cover", "total_gain", "total_cover"]
    *
    * @return featureScoreMap  key: feature index, value: feature importance score
    */
  @throws(classOf[XGBoostError])
  def getScore(featureMap: String, importanceType: String): Map[String, Double] = {
    Map(booster.getScore(featureMap, importanceType)
        .asScala.mapValues(_.doubleValue).toSeq: _*)
  }

  /**
    * Get importance of each feature based on information gain or cover
    * , with specified feature names.
    * Supported: ["gain, "cover", "total_gain", "total_cover"]
    *
    * @return featureScoreMap  key: feature name, value: feature importance score
    */
  @throws(classOf[XGBoostError])
  def getScore(featureNames: Array[String], importanceType: String): Map[String, Double] = {
    Map(booster.getScore(featureNames, importanceType)
        .asScala.mapValues(_.doubleValue).toSeq: _*)
  }

  /**
    * Get the number of model features.
    *
    * @return number of features
    */
  @throws(classOf[XGBoostError])
  def getNumFeature: Long = booster.getNumFeature

  def getVersion: Int = booster.getVersion

  /**
    * Save model into a raw byte array.  Available options are "json", "ubj" and "deprecated".
    */
  @throws(classOf[XGBoostError])
  def toByteArray(format: String): Array[Byte] = {
    booster.toByteArray(format)
  }

  /**
    * Save model into a raw byte array. Currently it's using the deprecated format as
   *  default, which will be changed into `ubj` in future releases.
    */
  @throws(classOf[XGBoostError])
  def toByteArray: Array[Byte] = {
    booster.toByteArray()
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
