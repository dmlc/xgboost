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

import ml.dmlc.xgboost4j.java.XGBoostError
import scala.collection.mutable

trait Booster extends Serializable {


  /**
   * set parameter
   *
   * @param key   param name
   * @param value param value
   */
  @throws(classOf[XGBoostError])
  def setParam(key: String, value: String)

  /**
   * set parameters
   *
   * @param params parameters key-value map
   */
  @throws(classOf[XGBoostError])
  def setParams(params: Map[String, AnyRef])

  /**
   * Update (one iteration)
   *
   * @param dtrain training data
   * @param iter   current iteration number
   */
  @throws(classOf[XGBoostError])
  def update(dtrain: DMatrix, iter: Int)

  /**
   * update with customize obj func
   *
   * @param dtrain training data
   * @param obj    customized objective class
   */
  @throws(classOf[XGBoostError])
  def update(dtrain: DMatrix, obj: ObjectiveTrait)

  /**
   * update with give grad and hess
   *
   * @param dtrain training data
   * @param grad   first order of gradient
   * @param hess   seconde order of gradient
   */
  @throws(classOf[XGBoostError])
  def boost(dtrain: DMatrix, grad: Array[Float], hess: Array[Float])

  /**
   * evaluate with given dmatrixs.
   *
   * @param evalMatrixs dmatrixs for evaluation
   * @param evalNames   name for eval dmatrixs, used for check results
   * @param iter        current eval iteration
   * @return eval information
   */
  @throws(classOf[XGBoostError])
  def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], iter: Int): String

  /**
   * evaluate with given customized Evaluation class
   *
   * @param evalMatrixs evaluation matrix
   * @param evalNames   evaluation names
   * @param eval        custom evaluator
   * @return eval information
   */
  @throws(classOf[XGBoostError])
  def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], eval: EvalTrait): String

  /**
   * Predict with data
   *
   * @param data dmatrix storing the input
   * @return predict result
   */
  @throws(classOf[XGBoostError])
  def predict(data: DMatrix): Array[Array[Float]]

  /**
   * Predict with data
   *
   * @param data         dmatrix storing the input
   * @param outPutMargin Whether to output the raw untransformed margin value.
   * @return predict result
   */
  @throws(classOf[XGBoostError])
  def predict(data: DMatrix, outPutMargin: Boolean): Array[Array[Float]]

  /**
   * Predict with data
   *
   * @param data         dmatrix storing the input
   * @param outPutMargin Whether to output the raw untransformed margin value.
   * @param treeLimit    Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @return predict result
   */
  @throws(classOf[XGBoostError])
  def predict(data: DMatrix, outPutMargin: Boolean, treeLimit: Int): Array[Array[Float]]

  /**
   * Predict with data
   *
   * @param data      dmatrix storing the input
   * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @param predLeaf  When this option is on, the output will be a matrix of (nsample, ntrees),
   *                  nsample = data.numRow with each record indicating the predicted leaf index of
   *                  each sample in each tree. Note that the leaf index of a tree is unique per
   *                  tree, so you may find leaf 1 in both tree 1 and tree 0.
   * @return predict result
   * @throws XGBoostError native error
   */
  @throws(classOf[XGBoostError])
  def predict(data: DMatrix, treeLimit: Int, predLeaf: Boolean): Array[Array[Float]]

  /**
   * save model to modelPath
   *
   * @param modelPath model path
   */
  @throws(classOf[XGBoostError])
  def saveModel(modelPath: String)

  /**
   * Dump model into a text file.
   *
   * @param modelPath file to save dumped model info
   * @param withStats bool Controls whether the split statistics are output.
   */
  @throws(classOf[IOException])
  @throws(classOf[XGBoostError])
  def dumpModel(modelPath: String, withStats: Boolean)

  /**
   * Dump model into a text file.
   *
   * @param modelPath  file to save dumped model info
   * @param featureMap featureMap file
   * @param withStats  bool
   *                   Controls whether the split statistics are output.
   */
  @throws(classOf[IOException])
  @throws(classOf[XGBoostError])
  def dumpModel(modelPath: String, featureMap: String, withStats: Boolean)

  /**
   * get importance of each feature
   *
   * @return featureMap  key: feature index, value: feature importance score
   */
  @throws(classOf[XGBoostError])
  def getFeatureScore: mutable.Map[String, Integer]

  /**
   * get importance of each feature
   *
   * @param featureMap file to save dumped model info
   * @return featureMap  key: feature index, value: feature importance score
   */
  @throws(classOf[XGBoostError])
  def getFeatureScore(featureMap: String): mutable.Map[String, Integer]

  def dispose
}
