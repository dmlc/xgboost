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

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

/**
  * A class which allows user to save checkpoints every a few rounds. If a previous job fails,
  * the job can restart training from a saved checkpoints instead of from scratch. This class
  * provides interface and helper methods for the checkpoint functionality.
  *
  * NOTE: This checkpoint is different from Rabit checkpoint. Rabit checkpoint is a native-level
  * checkpoint stored in executor memory. This is a checkpoint which Spark driver store on HDFS
  * for every a few iterations.
  *
  * @param sc the sparkContext object
  * @param checkpointPath the hdfs path to store checkpoints
  */
private[spark] class CheckpointManager(sc: SparkContext, checkpointPath: String) {
  private val logger = LogFactory.getLog("XGBoostSpark")
  private val modelSuffix = ".model"

  private def getPath(version: Int) = {
    s"$checkpointPath/$version$modelSuffix"
  }

  private def getExistingVersions: Seq[Int] = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (checkpointPath.isEmpty || !fs.exists(new Path(checkpointPath))) {
      Seq()
    } else {
      fs.listStatus(new Path(checkpointPath)).map(_.getPath.getName).collect {
        case fileName if fileName.endsWith(modelSuffix) => fileName.stripSuffix(modelSuffix).toInt
      }
    }
  }

  def cleanPath(): Unit = {
    if (checkpointPath != "") {
      FileSystem.get(sc.hadoopConfiguration).delete(new Path(checkpointPath), true)
    }
  }

  /**
    * Load existing checkpoint with the highest version as a Booster object
    *
    * @return the booster with the highest version, null if no checkpoints available.
    */
  private[spark] def loadCheckpointAsBooster: Booster = {
    val versions = getExistingVersions
    if (versions.nonEmpty) {
      val version = versions.max
      val fullPath = getPath(version)
      val inputStream = FileSystem.get(sc.hadoopConfiguration).open(new Path(fullPath))
      logger.info(s"Start training from previous booster at $fullPath")
      val booster = SXGBoost.loadModel(inputStream)
      booster.booster.setVersion(version)
      booster
    } else {
      null
    }
  }

  /**
    * Clean up all previous checkpoints and save a new checkpoint
    *
    * @param checkpoint the checkpoint to save as an XGBoostModel
    */
  private[spark] def updateCheckpoint(checkpoint: Booster): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val prevModelPaths = getExistingVersions.map(version => new Path(getPath(version)))
    val fullPath = getPath(checkpoint.getVersion)
    val outputStream = fs.create(new Path(fullPath), true)
    logger.info(s"Saving checkpoint model with version ${checkpoint.getVersion} to $fullPath")
    checkpoint.saveModel(outputStream)
    prevModelPaths.foreach(path => fs.delete(path, true))
  }

  /**
    * Clean up checkpoint boosters with version higher than or equal to the round.
    *
    * @param round the number of rounds in the current training job
    */
  private[spark] def cleanUpHigherVersions(round: Int): Unit = {
    val higherVersions = getExistingVersions.filter(_ / 2 >= round)
    higherVersions.foreach { version =>
      val fs = FileSystem.get(sc.hadoopConfiguration)
      fs.delete(new Path(getPath(version)), true)
    }
  }

  /**
    * Calculate a list of checkpoint rounds to save checkpoints based on the checkpointInterval
    * and total number of rounds for the training. Concretely, the checkpoint rounds start with
    * prevRounds + checkpointInterval, and increase by checkpointInterval in each step until it
    * reaches total number of rounds. If checkpointInterval is 0, the checkpoint will be disabled
    * and the method returns Seq(round)
    *
    * @param checkpointInterval Period (in iterations) between checkpoints.
    * @param round the total number of rounds for the training
    * @return a seq of integers, each represent the index of round to save the checkpoints
    */
  private[spark] def getCheckpointRounds(checkpointInterval: Int, round: Int): Seq[Int] = {
    if (checkpointPath.nonEmpty && checkpointInterval > 0) {
      val prevRounds = getExistingVersions.map(_ / 2)
      val firstCheckpointRound = (0 +: prevRounds).max + checkpointInterval
      (firstCheckpointRound until round by checkpointInterval) :+ round
    } else if (checkpointInterval <= 0) {
      Seq(round)
    } else {
      throw new IllegalArgumentException("parameters \"checkpoint_path\" should also be set.")
    }
  }
}

object CheckpointManager {

  case class CheckpointParam(
      checkpointPath: String,
      checkpointInterval: Int,
      skipCleanCheckpoint: Boolean)

  private[spark] def extractParams(params: Map[String, Any]): CheckpointParam = {
    val checkpointPath: String = params.get("checkpoint_path") match {
      case None => ""
      case Some(path: String) => path
      case _ => throw new IllegalArgumentException("parameter \"checkpoint_path\" must be" +
        " an instance of String.")
    }

    val checkpointInterval: Int = params.get("checkpoint_interval") match {
      case None => 0
      case Some(freq: Int) => freq
      case _ => throw new IllegalArgumentException("parameter \"checkpoint_interval\" must be" +
        " an instance of Int.")
    }

    val skipCheckpointFile: Boolean = params.get("skip_clean_checkpoint") match {
      case None => false
      case Some(skipCleanCheckpoint: Boolean) => skipCleanCheckpoint
      case _ => throw new IllegalArgumentException("parameter \"skip_clean_checkpoint\" must be" +
        " an instance of Boolean")
    }
    CheckpointParam(checkpointPath, checkpointInterval, skipCheckpointFile)
  }
}
