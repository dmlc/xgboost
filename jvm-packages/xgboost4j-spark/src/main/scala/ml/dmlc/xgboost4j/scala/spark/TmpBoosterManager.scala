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
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

/**
  * A class which allows user to save temporary boosters every a few rounds. If a previous job
  * fails, the job can restart training from a saved booster instead of from scratch. This class
  * provides interface and helper methods for the tmp booster functionality.
  *
  * @param sc the sparkContext object
  * @param boosterTmpPath the hdfs path to store temporary boosters
  */
class TmpBoosterManager(sc: SparkContext, boosterTmpPath: String) {
  private val logger = LogFactory.getLog("XGBoostSpark")
  private val modelSuffix = ".model"

  private def getPath(version: Int) = {
    s"$boosterTmpPath/$version$modelSuffix"
  }

  private def getExistingVersions: Seq[Int] = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (boosterTmpPath.isEmpty || !fs.exists(new Path(boosterTmpPath))) {
      Seq()
    } else {
      fs.listStatus(new Path(boosterTmpPath)).map(_.getPath.getName).collect {
        case fileName if fileName.endsWith(modelSuffix) => fileName.stripSuffix(modelSuffix).toInt
      }
    }
  }

  /**
    * Load existing booster with the highest version.
    *
    * @return the booster with the highest version.
    */
  private[spark] def loadBooster: Booster = {
    val versions = getExistingVersions
    if (versions.nonEmpty) {
      val version = versions.max
      val fullPath = getPath(version)
      logger.info(s"Start training from previous booster at $fullPath")
      val model = XGBoost.loadModelFromHadoopFile(fullPath)(sc)
      model.booster.booster.setVersion(version)
      model.booster
    } else {
      null
    }
  }

  /**
    * Clean up all previous models and save a new model
    *
    * @param model the xgboost model to save
    */
  private[spark] def updateModel(model: XGBoostModel): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val prevModelPaths = getExistingVersions.map(version => new Path(getPath(version)))
    val fullPath = getPath(model.version)
    logger.info(s"Saving temporary model with version ${model.version} to $fullPath")
    model.saveModelAsHadoopFile(fullPath)(sc)
    prevModelPaths.foreach(path => fs.delete(path, true))
  }

  /**
    * Clean up tmp boosters with version higher than or equal to the round.
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
    * Calculate a list of checkpoint rounds to save tmp boosters based on the savingFreq and
    * total number of rounds for the training. Concretely, the saving rounds start with
    * prevRounds + savingFreq, and increase by savingFreq in each step until it reaches total
    * number of rounds
    *
    * @param savingFreq the increase on rounds during each step of training
    * @param round the total number of rounds for the training
    * @return a seq of integers, each represent the index of round to save the tmp booster
    */
  private[spark] def getSavingRounds(savingFreq: Int, round: Int): Seq[Int] = {
    if (boosterTmpPath.nonEmpty && savingFreq > 0) {
      val prevRounds = getExistingVersions.map(_ / 2)
      val firstSavingRound = (0 +: prevRounds).max + savingFreq
      (firstSavingRound until round by savingFreq) :+ round
    } else if (savingFreq <= 0) {
      Seq(round)
    } else {
      throw new IllegalArgumentException("parameters \"booster_tmp_path\" should also be set.")
    }
  }
}

object TmpBoosterManager {

  private[spark] def extractParams(params: Map[String, Any]): (String, Int) = {
    val boosterTmpPath: String = params.get("booster_tmp_path") match {
      case None => ""
      case Some(path: String) => path
      case _ => throw new IllegalArgumentException("parameter \"booster_tmp_path\" must be" +
        " an instance of String.")
    }

    val savingFreq: Int = params.get("saving_frequency") match {
      case None => 0
      case Some(freq: Int) => freq
      case _ => throw new IllegalArgumentException("parameter \"saving_frequency\" must be" +
        " an instance of Int.")
    }
    (boosterTmpPath, savingFreq)
  }
}
