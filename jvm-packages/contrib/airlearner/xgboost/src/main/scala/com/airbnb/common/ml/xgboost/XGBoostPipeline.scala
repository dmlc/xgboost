package com.airbnb.common.ml.xgboost

import java.io.File

import scala.language.postfixOps
import scala.util.Try

import com.typesafe.config.Config
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel

import com.airbnb.common.ml.search.{MonteCarloParams, MonteCarloSearch}
import com.airbnb.common.ml.util.{HDFSUtil, PipelineUtil, ScalaLogging}
import com.airbnb.common.ml.xgboost.config.XGBoostSearchConfig
import com.airbnb.common.ml.xgboost.data._
import com.airbnb.common.ml.xgboost.model.XGBoostModel


object XGBoostPipeline extends ScalaLogging {

  val minLoss          = 0.2
  val minDiff          = 0.000001
  val trainFile        = "train"
  val evalFile         = "eval"
  val roundIdx         = 4
  val defaultTmpFolder = "/mnt/var/spark/tmp/"

  def trainWithSearchedParamsAndPreparedData(
      sc: SparkContext,
      config: Config
  ): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val output = config.getString("model_output")
    val tmpFolder: String = Try(config.getString("tmp_folder"))
      .getOrElse(defaultTmpFolder)
    HDFSUtil.createPath(output)
    val overwrite = config.getBoolean("overwrite")

    val paramData = ModelData.getParams(sc, config.getString("params_query"))
    paramData.map {
      case (id, params) => {
        val param = params.head
        val paramMap = getParamMap(param)
        val round = param(roundIdx).toInt
        val modelPath = output + id
        val hfs = FileSystem.get(new java.net.URI(modelPath), new Configuration())
        val destPath = new Path(modelPath)
        if (overwrite || !hfs.exists(destPath)) {
          trainAndSaveModelByID(
            id,
            round,
            cfg.stableParamMap,
            cfg.trainingDataDir,
            output,
            paramMap,
            tmpFolder
          )
        }
      }
    }.count()
  }

  def trainWithSearchedParams(sc: SparkContext, config: Config): Unit = {
    // number of iterations
    val output = config.getString("model_output")

    // read training data, available at xgboost/demo/data
    val trainData = ModelData.getLabeledPoints(
      sc,
      config.getString("training_query"),
      BaseTrainingModelData)

    val paramData = ModelData.getParams(sc, config.getString("params_query"))

    trainData.join(paramData)
      .map {
        case (id, (data, params)) => {
          val param = params.head
          val training = new DMatrix(data.iterator, null)
          val paramMap = getParamMap(param)
          // order of params should be consistent with params order in search.conf
          // round is No.6 in params so index is 5
          val round = param(roundIdx).toInt
          logger.info(paramMap.toString())
          XGBoostModel.trainAndSave(training, output + id, round, paramMap)
        }
      }.count() // this is empty operation just for spark to run
  }

  // order of params should be consistent with params order in search.conf
  def getParamMap(params: Array[Double]): Map[String, Any] = {
    List(
      "gamma" -> params(0),
      "eta" -> params(1),
      "colsample_bytree" -> params(2),
      "lambda" -> params(3),
      "min_child_weight" -> params(5).toInt,
      "max_depth" -> params(6).toInt,
      "objective" -> "binary:logistic").toMap
  }

  def train(sc: SparkContext, config: Config): Unit = {
    // read training data, available at xgboost/demo/data
    val trainData = ModelData.getLabeledPoints(
      sc,
      config.getString("training_query"),
      BaseTrainingModelData)
    // define parameters
    val paramMap = List(
      "gamma" -> 0.8,
      "eta" -> 0.3,
      "max_depth" -> 13,
      "silent" -> 1,
      "objective" -> "binary:logistic").toMap

    // number of iterations
    val round = config.getInt("round")
    val output = config.getString("model_output")

    trainData.map {
      case (id, data) => {
        val training = new DMatrix(data.iterator, null)
        XGBoostModel.trainAndSave(training, output + id, round, paramMap)
      }
    }.count() // this is empty operation just for spark to run
  }

  def saveData(
      t: Seq[LabeledPoint],
      path: String,
      tmpFolder: String
  ): Unit = {
    val matrix = new DMatrix(t.iterator, null)
    val localTmpFile = Utils.generateTmpFile(tmpFolder)

    // make sure files are removed in case of exception
    new File(localTmpFile).deleteOnExit()

    // save may fail due to no space.
    // we should let it fail and regenerate data.
    matrix.saveBinary(localTmpFile)
    logger.info(s"xgboostUploadData $localTmpFile $path")
    val hfs = FileSystem.get(new java.net.URI(path), new Configuration())
    // same as hdfs dfs -copyFromLocal -f,
    // src file removed upon finish
    hfs.copyFromLocalFile(true, new Path(localTmpFile), new Path(path))
    matrix.delete()
  }

  private def getDataFromHive(sc: SparkContext, cfg: XGBoostSearchConfig) = {
    val trainingData = ModelData.getLabeledPoints(
      sc, cfg.trainingData, BaseTrainingModelData)
    val evalData = ModelData.getLabeledPoints(
      sc, cfg.evalData, BaseTrainingModelData)
    trainingData.join(evalData)
  }

  def prepareTrainingData(sc: SparkContext, config: Config): Unit = {
    prepareData(sc, config, trainFile)
  }

  def prepareEvalData(sc: SparkContext, config: Config): Unit = {
    prepareData(sc, config, evalFile)
  }

  def prepareData(
      sc: SparkContext,
      config: Config,
      fileName: String
  ): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val (query, dir) = if (fileName == trainFile) {
      HDFSUtil.createPath(cfg.trainingDataDir)
      (cfg.trainingData, cfg.trainingDataDir)
    } else {
      HDFSUtil.createPath(cfg.evalDataDir)
      (cfg.evalData, cfg.evalDataDir)
    }

    val data = ModelData.getLabeledPoints(sc, query, BaseTrainingModelData)

    val number = data.map {
      case (id, d) => {
        saveData(d, Utils.getHDFSDataFile(dir, id, fileName), cfg.tmpFolder)
      }
    }
      .count()
    logger.info(s"prepareData $number")
  }

  def getDataFromHDFS(
      dataDir: String,
      id: String,
      dataFile: String,
      tmpFolder: String
  ): String = {
    val hfs = FileSystem.get(new java.net.URI(dataDir), new Configuration())
    val localTmpFile = Utils.generateTmpFile(tmpFolder)
    val hdfsFile = Utils.getHDFSDataFile(dataDir, id, dataFile)
    logger.info(s"xgboostDownloadData $localTmpFile $hdfsFile")
    // same as hdfs dfs -copyToLocal
    hfs.copyToLocalFile(new Path(hdfsFile), new Path(localTmpFile))
    // make sure file will be remove even executor preempted by spark
    new File(localTmpFile).deleteOnExit()
    localTmpFile
  }

  def trainAndSaveModelByID(
      id: String,
      round: Int,
      stableParamMap: List[(String, Any)],
      dataPath: String,
      modelPath: String,
      currentParams: Map[String, Any],
      tmpFolder: String
  ): Unit = {
    var localTrainData = ""

    try {
      localTrainData = getDataFromHDFS(dataPath, id, trainFile, tmpFolder)
      val training = new DMatrix(localTrainData)

      val params = (currentParams.toList ::: stableParamMap).toMap

      XGBoostModel.trainAndSave(training, modelPath + id, round, params)
      // not catching errors to make sure it failed so spark retry training.
    }
    finally {
      // del on exit is not enough since one executor may run multiple files in its life
      Utils.delFile(localTrainData)
    }
  }

  def searchByID(
      id: String,
      cfg: XGBoostSearchConfig
  ): (String, Double) = {
    var localTrainFata = ""
    var localEvalFata = ""
    try {
      localTrainFata = getDataFromHDFS(cfg.trainingDataDir, id, trainFile, cfg.tmpFolder)
      localEvalFata = getDataFromHDFS(cfg.evalDataDir, id, evalFile, cfg.tmpFolder)

      val model = XGBoostModel.getModelByFile(localTrainFata, localEvalFata)
      val (param, loss) = MonteCarloSearch.run(
        model,
        cfg.paramMap,
        cfg.stableParamMap,
        cfg.minRound,
        minDiff,
        minLoss)
      val output = s"$id\t$param\t$loss"
      logger.info(s"xgboost output $output")
      (param, loss)
    }
    catch {
      case a: Throwable => {
        logger.info(a.getLocalizedMessage)
        // give a fake result so that the process continue
        ("1.0", 100)
      }
    }
    finally {
      // del on exit is not enough since one executor may run multiple files in its life
      Utils.delFile(localTrainFata)
      Utils.delFile(localEvalFata)
    }
  }

  def saveSearchedParam[U](
      sc: SparkContext,
      cfg: XGBoostSearchConfig,
      params: RDD[U],
      id: String
  ): Unit = {
    val hc = new HiveContext(sc)
    val outputPath = if (cfg.useModelPostfix) {
      cfg.outputPath.replace("$MODEL_POSTFIX", id)
    } else {
      cfg.outputPath.replace("$MODEL_POSTFIX", "")
    }
    val partitionSpec = if (cfg.useModelPostfix) {
      cfg.partitionSpec.replace("$MODEL_POSTFIX", id)
    } else {
      cfg.partitionSpec.replace("$MODEL_POSTFIX", "")
    }
    PipelineUtil.saveToHdfsAndUpdateHive(
      hc, outputPath, cfg.outputTable, partitionSpec, params, cfg.overwrite)
  }

  def saveSingleSearchByID(sc: SparkContext, id: String, cfg: XGBoostSearchConfig): Unit = {
    // only use first element since this is single ParamSearch
    val rdd = sc.parallelize(List.range(0, cfg.round, 1))
    val best = rdd.map(d => {
      searchByID(id, cfg)
    }).reduce((a, b) => {
      if (a._2 < b._2) {
        a
      } else {
        b
      }
    })
    val output = s"$id\t${best._1}\t${best._2}"
    val save = sc.parallelize(List(output))
    saveSearchedParam(sc, cfg, save, id)
  }

  // only search params for one kdt node, so that we know the range of params
  // and speed up the params search for all areas
  // it loads training data from the output of prepareData
  def singleParamSearch(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val id = config.getString("id")
    saveSingleSearchByID(sc, id, cfg)
  }

  // search each node separately
  def paramSearchWithPreparedDataByNode(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)

    val rawData = ModelData.getDataFrame(sc, config.getString("node_query")).map(
      row => {
        row.getAs[String](0)
      }
    ).collect()
    val data = if (cfg.reverse) {
      rawData.reverse
    } else {
      rawData
    }

    for (id <- data) {
      saveSingleSearchByID(sc, id, cfg)
    }
  }

  // search all node at the same time
  def paramSearchWithPreparedData(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val data = ModelData.getDataFrame(sc, config.getString("node_query")).map(
      row => {
        row.getAs[String](0)
      }
    ).cache()
    var lastBest: Option[RDD[(String, (String, Double))]] = None
    for (a <- 1 to cfg.round) {
      val currentBest = data.map(
        id => {
          (id, searchByID(id, cfg))
        }
      )
      val best = getBestParams(lastBest, currentBest)
      lastBest = Some(best)
      // convert to string
      val output = best.map {
        case (id, (param, loss)) => {
          s"$id\t$param\t$loss"
        }
      }
      saveSearchedParam(sc, cfg, output, "")
    }
  }

  // this is simlar to paramSearchWithPrepareData, with flatmap
  // it is possible to use more executors when clusters are free
  def paramFlatSearchWithPreparedData(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val rawData = ModelData.getDataFrame(sc, config.getString("node_query")).map(
      row => {
        row.getAs[String](0)
      }
    ).flatMap(c => List.fill(cfg.round)(c))
      .collect().toList

    val shuffledData = scala.util.Random.shuffle(rawData)

    // spark cluster without collect spark cluster thinks this is a small job only assign 8
    // executors.
    val data = sc.parallelize(shuffledData)

    val best = data.map(
      id => {
        (id, searchByID(id, cfg))
      }
    ).reduceByKey(bestLoss).map {
      case (id, (param, loss)) => {
        s"$id\t$param\t$loss"
      }
    }
    saveSearchedParam(sc, cfg, best, "")
  }

  def bestLoss(a: (String, Double), b: (String, Double)): (String, Double) = {
    if (a._2 < b._2) {
      a
    } else {
      b
    }
  }

  def paramFlatSearchWithPreparedDataAndLastParam(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)
    val searchRatio = config.getDouble("search_ratio")
    val paramData = ModelData.getParams(sc, config.getString("params_query"))
      .map {
        case (id, params) => {
          val paramMap = MonteCarloParams.adjustParamsFromPreviousSearch(
            cfg.paramMap, params.head.toList, searchRatio)
          (id, paramMap)
        }
      }
      .flatMap(c => List.fill(cfg.round)(c))
      .collect().toList

    val shuffledData = scala.util.Random.shuffle(paramData)

    // spark cluster without collect spark cluster thinks this is a small job only assign 8
    // executors.
    val data = sc.parallelize(shuffledData)

    val best = data.map {
      case (id, params) => {
        val newConfig = XGBoostSearchConfig.updateParamMap(cfg, params)
        (id, searchByID(id, newConfig))
      }
    }
      .reduceByKey(bestLoss).map {
      case (id, (param, loss)) => {
        s"$id\t$param\t$loss"
      }
    }
    saveSearchedParam(sc, cfg, best, config.getString("next_model_postfix"))
  }

  def paramSearch(sc: SparkContext, config: Config): Unit = {
    val cfg = XGBoostSearchConfig.loadConfig(sc, config)

    val data = getDataFromHive(sc, cfg).persist(StorageLevel.MEMORY_AND_DISK)
    var lastBest: Option[RDD[(String, (String, Double))]] = None
    for (a <- 1 to cfg.round) {
      val currentBest = data.map {
        case (id, (t, e)) => {
          logger.info(s"paramSearch $id")
          val model = XGBoostModel.getModelByLabeledPoint(t, e)
          val (param, loss) = MonteCarloSearch.run(
            model,
            cfg.paramMap,
            cfg.stableParamMap,
            cfg.minRound,
            minDiff,
            minLoss)
          (id, (param, loss))
        }
      }
      val best = getBestParams(lastBest, currentBest)
      lastBest = Some(best)
      // convert to string
      val output = best.map {
        case (id, (param, loss)) => {
          s"$id\t$param\t$loss"
        }
      }
      saveSearchedParam(sc, cfg, output, "")
    }
  }

  def getBestParams(
      lastBest: Option[RDD[(String, (String, Double))]],
      currentBest: RDD[(String, (String, Double))]
  ): RDD[(String, (String, Double))] = {
    if (lastBest.isDefined) {
      val last = lastBest.get
      val joinedBest = last.join(currentBest).map {
        case (id, (p1, p2)) => {
          if (p1._2 < p2._2) {
            (id, p1)
          } else {
            (id, p2)
          }
        }
      }
      last.unpersist()
      joinedBest.cache()
    } else {
      currentBest.cache()
    }
  }
}
