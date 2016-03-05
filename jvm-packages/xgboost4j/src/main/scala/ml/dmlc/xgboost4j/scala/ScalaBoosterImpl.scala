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

import ml.dmlc.xgboost4j.java
import scala.collection.JavaConverters._
import scala.collection.mutable

private[scala] class ScalaBoosterImpl private[xgboost4j](booster: java.Booster) extends Booster {

  override def setParam(key: String, value: String): Unit = {
    booster.setParam(key, value)
  }

  override def update(dtrain: DMatrix, iter: Int): Unit = {
    booster.update(dtrain.jDMatrix, iter)
  }

  override def update(dtrain: DMatrix, obj: ObjectiveTrait): Unit = {
    booster.update(dtrain.jDMatrix, obj)
  }

  override def dumpModel(modelPath: String, withStats: Boolean): Unit = {
    booster.dumpModel(modelPath, withStats)
  }

  override def dumpModel(modelPath: String, featureMap: String, withStats: Boolean): Unit = {
    booster.dumpModel(modelPath, featureMap, withStats)
  }

  override def setParams(params: Map[String, AnyRef]): Unit = {
    booster.setParams(params.asJava)
  }

  override def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], iter: Int): String = {
    booster.evalSet(evalMatrixs.map(_.jDMatrix), evalNames, iter)
  }

  override def evalSet(evalMatrixs: Array[DMatrix], evalNames: Array[String], eval: EvalTrait):
      String = {
    booster.evalSet(evalMatrixs.map(_.jDMatrix), evalNames, eval)
  }

  override def dispose: Unit = {
    booster.dispose()
  }

  override def predict(data: DMatrix): Array[Array[Float]] = {
    booster.predict(data.jDMatrix)
  }

  override def predict(data: DMatrix, outPutMargin: Boolean): Array[Array[Float]] = {
    booster.predict(data.jDMatrix, outPutMargin)
  }

  override def predict(data: DMatrix, outPutMargin: Boolean, treeLimit: Int):
      Array[Array[Float]] = {
    booster.predict(data.jDMatrix, outPutMargin, treeLimit)
  }

  override def predict(data: DMatrix, treeLimit: Int, predLeaf: Boolean): Array[Array[Float]] = {
    booster.predict(data.jDMatrix, treeLimit, predLeaf)
  }

  override def boost(dtrain: DMatrix, grad: Array[Float], hess: Array[Float]): Unit = {
    booster.boost(dtrain.jDMatrix, grad, hess)
  }

  override def getFeatureScore: mutable.Map[String, Integer] = {
    booster.getFeatureScore.asScala
  }

  override def getFeatureScore(featureMap: String): mutable.Map[String, Integer] = {
    booster.getFeatureScore(featureMap).asScala
  }

  override def saveModel(modelPath: String): Unit = {
    booster.saveModel(modelPath)
  }

  override def finalize(): Unit = {
    super.finalize()
    dispose
  }
}
