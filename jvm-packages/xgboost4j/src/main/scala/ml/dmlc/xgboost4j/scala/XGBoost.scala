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

import java.io.InputStream
import ml.dmlc.xgboost4j.java.{XGBoostError, XGBoost => JXGBoost}

import scala.jdk.CollectionConverters._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

/**
  * XGBoost Scala Training function.
  */
object XGBoost {

  private[scala] def trainAndSaveCheckpoint(
      dtrain: DMatrix,
      params: Map[String, Any],
      numRounds: Int,
      watches: Map[String, DMatrix] = Map(),
      metrics: Array[Array[Float]] = null,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      earlyStoppingRound: Int = 0,
      prevBooster: Booster,
      checkpointParams: Option[ExternalCheckpointParams]): Booster = {

    // we have to filter null value for customized obj and eval
    val jParams: java.util.Map[String, AnyRef] =
      params.filter(_._2 != null).mapValues(_.toString.asInstanceOf[AnyRef]).toMap.asJava

    val jWatches = watches.mapValues(_.jDMatrix).toMap.asJava
    val jBooster = if (prevBooster == null) {
      null
    } else {
      prevBooster.booster
    }

    val xgboostInJava = checkpointParams.
      map(cp => {
          JXGBoost.trainAndSaveCheckpoint(
            dtrain.jDMatrix,
            jParams,
            numRounds, jWatches, metrics, obj, eval, earlyStoppingRound, jBooster,
            cp.checkpointInterval,
            cp.checkpointPath,
            new Path(cp.checkpointPath).getFileSystem(new Configuration()))
        }).
      getOrElse(
        JXGBoost.train(
          dtrain.jDMatrix,
          jParams,
          numRounds, jWatches, metrics, obj, eval, earlyStoppingRound, jBooster)
      )
    if (prevBooster == null) {
      new Booster(xgboostInJava)
    } else {
      // Avoid creating a new SBooster with the same JBooster
      prevBooster
    }
  }

  /**
    * Train a booster given parameters.
    *
    * @param dtrain  Data to be trained.
    * @param params  Parameters.
    * @param round   Number of boosting iterations.
    * @param watches a group of items to be evaluated during training, this allows user to watch
    *                performance on the validation set.
    * @param metrics array containing the evaluation metrics for each matrix in watches for each
    *                iteration
    * @param earlyStoppingRound if non-zero, training would be stopped
    *                           after a specified number of consecutive
    *                           increases in any evaluation metric.
    * @param obj     customized objective
    * @param eval    customized evaluation
    * @param booster train from scratch if set to null; train from an existing booster if not null.
    * @return The trained booster.
    */
  @throws(classOf[XGBoostError])
  def train(
      dtrain: DMatrix,
      params: Map[String, Any],
      round: Int,
      watches: Map[String, DMatrix] = Map(),
      metrics: Array[Array[Float]] = null,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      earlyStoppingRound: Int = 0,
      booster: Booster = null): Booster = {
    trainAndSaveCheckpoint(dtrain, params, round, watches, metrics, obj, eval, earlyStoppingRound,
      booster, None)
  }

  /**
    * Cross-validation with given parameters.
    *
    * @param data    Data to be trained.
    * @param params  Booster params.
    * @param round   Number of boosting iterations.
    * @param nfold   Number of folds in CV.
    * @param metrics Evaluation metrics to be watched in CV.
    * @param obj     customized objective
    * @param eval    customized evaluation
    * @return evaluation history
    */
  @throws(classOf[XGBoostError])
  def crossValidation(
      data: DMatrix,
      params: Map[String, Any],
      round: Int,
      nfold: Int = 5,
      metrics: Array[String] = null,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null): Array[String] = {
    JXGBoost.crossValidation(
      data.jDMatrix, params.map{ case (key: String, value) => (key, value.toString)}.
        toMap[String, AnyRef].asJava,
      round, nfold, metrics, obj, eval)
  }

  /**
    * load model from modelPath
    *
    * @param modelPath booster modelPath
    */
  @throws(classOf[XGBoostError])
  def loadModel(modelPath: String): Booster = {
    val xgboostInJava = JXGBoost.loadModel(modelPath)
    new Booster(xgboostInJava)
  }

  /**
    * Load a new Booster model from a file opened as input stream.
    * The assumption is the input stream only contains one XGBoost Model.
    * This can be used to load existing booster models saved by other XGBoost bindings.
    *
    * @param in The input stream of the file.
    * @return The create booster
    */
  @throws(classOf[XGBoostError])
  def loadModel(in: InputStream): Booster = {
    val xgboostInJava = JXGBoost.loadModel(in)
    new Booster(xgboostInJava)
  }
}

private[scala] case class ExternalCheckpointParams(
    checkpointInterval: Int,
    checkpointPath: String,
    skipCleanCheckpoint: Boolean)

private[scala] object ExternalCheckpointParams {

  def extractParams(params: Map[String, Any]): Option[ExternalCheckpointParams] = {
    val checkpointPath: String = params.get("checkpoint_path") match {
      case None | Some(null) | Some("") => null
      case Some(path: String) => path
      case _ => throw new IllegalArgumentException("parameter \"checkpoint_path\" must be" +
        s" an instance of String, but current value is ${params("checkpoint_path")}")
    }

    val checkpointInterval: Int = params.get("checkpoint_interval") match {
      case None => 0
      case Some(freq: Int) => freq
      case _ => throw new IllegalArgumentException("parameter \"checkpoint_interval\" must be" +
        " an instance of Int.")
    }

    val skipCleanCheckpointFile: Boolean = params.get("skip_clean_checkpoint") match {
      case None => false
      case Some(skipCleanCheckpoint: Boolean) => skipCleanCheckpoint
      case _ => throw new IllegalArgumentException("parameter \"skip_clean_checkpoint\" must be" +
        " an instance of Boolean")
    }
    if (checkpointPath == null || checkpointInterval == 0) {
      None
    } else {
      Some(ExternalCheckpointParams(checkpointInterval, checkpointPath, skipCleanCheckpointFile))
    }
  }
}


