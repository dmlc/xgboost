package com.airbnb.common.ml.strategy.trainer

import scala.reflect.ClassTag
import scala.util.Random
import scala.util.Try

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel

import com.airbnb.common.ml.strategy.config.{BaseSearchConfig, DirectQueryEvalConfig, EvalConfig, TrainingConfig, TrainingOptions}
import com.airbnb.common.ml.strategy.data.DataLoadingRules
import com.airbnb.common.ml.strategy.data.{BinaryTrainingSample, ModelOutput, TrainingData}
import com.airbnb.common.ml.strategy.eval.BinaryMetrics
import com.airbnb.common.ml.strategy.params.StrategyParams
import com.airbnb.common.ml.util.{HDFSUtil, HiveUtil, PipelineUtil, RandomUtil, ScalaLogging, Sort}


trait BinaryRegressionTrainer[T <: BinaryTrainingSample]
  extends Serializable
    with ScalaLogging {

  def strategyParams: StrategyParams[T]

  def trainingDataType: TrainingData[T]

  def getDefaultParams(trainingOptions: TrainingOptions): StrategyParams[T] = {
    if (strategyParams.params.length == 0) {
      strategyParams.getDefaultParams(trainingOptions)
    } else {
      strategyParams
    }
  }

  /**
    * Given a query, load StrategyParams from Hive
    *
    * @param hiveContext HiveContext
    * @param paramsQuery Hive query to pull params
    * @return the loaded per-model params
    */
  def loadParamsFromHive(hiveContext: HiveContext, paramsQuery: String):
  RDD[(String, StrategyParams[T])] = {
    hiveContext.sql(paramsQuery)
      // params share same key with training data.
      .map(row => (trainingDataType.parseKeyFromHiveRow(row),
                    strategyParams.parseParamsFromHiveRow(row)))
  }

  def getLearningRate(
      r0: Double, r1: Double,
      example: T,
      options: TrainingOptions
  ): Double

  // each trainer provides its own schema to generate data frame
  def createDataFrameFromModelOutput(models: RDD[(String, StrategyParams[T])], hc: HiveContext):
  DataFrame

  def loadModelWithIndexMapFromHdfs(output: String):
  scala.collection.Map[(java.lang.String, Int), StrategyParams[T]] = {
    val modelStr = HDFSUtil.readStringFromFile(output)
    modelStr
      .split("\n")
      .map(strategyParams.parseLine)
      .toMap
  }

  // load Training data by join two DataFrame with key,
  // this generates new training samples which
  // should has same fields as the loadTrainingDataFromHive methods
  // use this method if you want to change any fields and retrain model
  def loadTrainingDataByJoinTwoDataFrames(
      main: DataFrame,
      key: Seq[String],
      addition: DataFrame
  )(implicit c: ClassTag[T]):
  RDD[(String, Seq[T])] = {
    val joined = main.join(addition, key)
    trainingDataType.loadDataFromDataFrame(joined)
  }

  // default is same as loadTrainingDataFromHive
  // allow to override so that it loads different data
  // refer to StrategyModelTrainerV1.loadDataWithOptions for how to use it
  // TODO retire it
  def loadDataWithOptions(
      config: Config,
      hc: HiveContext,
      dataQuery: String,
      forTraining: Boolean = true
  )(implicit c: ClassTag[T]): RDD[(String, Seq[T])] = {
    trainingDataType.loadDataFromHive(hc, dataQuery)
  }

  // return trained result in data frame,
  // i.e. StrategyModelTrainerV1 return StrategyModelDataSource.schema
  def getResultDataFrame(
      strategyConfig: Config,
      hc: HiveContext,
      trainingData: RDD[(String, Seq[T])]
  )(implicit c: ClassTag[T]): DataFrame = {
    val models: RDD[(String, StrategyParams[T])] =
      trainStrategyModelWithRDD(
        hc,
        strategyConfig,
        trainingData
      )
    createDataFrameFromModelOutput(models, hc)
  }

  /**
    * Given an RDD of [(id, TrainingData)], train strategy models for each set
    * of training datas and save the results to Hive.
    *
    * @param hc              HiveContext
    * @param config          top-level config
    * @param rawTrainingData (id -> [TrainingData])
    * @return RDD of (id, StrategyParams)
    */
  def trainStrategyModelWithRDD(
      hc: HiveContext,
      config: Config,
      rawTrainingData: RDD[(String, Seq[T])]
  )(implicit c: ClassTag[T]): RDD[(String, StrategyParams[T])] = {
    // Load training options from config
    val trainingOptions: TrainingOptions =
      TrainingOptions
        .loadBaseTrainingOptions(
          config.getConfig("training_options"))

    // Prepare the per-model strategy params for our training data
    val trainingExamples: RDD[(String, (Seq[T], Option[StrategyParams[T]]))] =
      joinTrainingSamplesWithParams(hc, config, rawTrainingData, trainingOptions)

    // Perform the training and return trained strategy params
    trainAll(trainingExamples, trainingOptions)
  }

  /**
    * support parameter search, i.e. given Array[TrainingOptions] return
    * trained models for each TrainingOption in the array key in the
    * returning RDD is (id, param_idx), where param_idx specifies
    * the index of the TrainingOptions used to run the training.
    *
    * @param examples             training samples
    * @param trainingOptionsArray a TrainingOption for each hyperparam combo
    * @return (id, training_options_index) -> trained_params
    */
  def trainAllWithOptionArray(
      examples: RDD[(String, Seq[T])],
      trainingOptionsArray: Array[TrainingOptions]
  ): RDD[((String, Int), StrategyParams[T])] = {
    // Note: This does not support initialization from existing parameters
    examples
      .mapPartitions(iterable => {
        iterable.flatMap { example =>
          val id: String = example._1
          val samples: Seq[T] = example._2

          // For each of the training options, train it using all
          // of the samples for this id instance
          trainingOptionsArray
            .zipWithIndex
            .map {
              case (option: TrainingOptions, idx: Int) => {
                ((id, idx), train(samples, option)._1)
              }
            }.iterator
        }
      })
  }

  /**
    * Given training examples and initial strategy params, train each of the models'
    * strategy params.
    *
    * @param examples training samples, with starting training options
    * @param options  training options
    * @return trained params
    */
  def trainAll(
      examples: RDD[(String, (Seq[T], Option[StrategyParams[T]]))],
      options: TrainingOptions
  ): RDD[(String, StrategyParams[T])] = {
    examples
      .map {
        case (id: String, (samples: Seq[T], param: Option[StrategyParams[T]])) => {
          (id, train(samples, options, param)._1)
        }
      }
  }

  def getParamsFromPerModelOptions(
      trainingExamples: RDD[(String, Seq[T])],
      perModelOptions: RDD[(String, Array[Double])],
      trainingOptions: TrainingOptions
  ): RDD[String] = {
    trainingExamples.
      leftOuterJoin(perModelOptions).
      map {
        case (id, (examples, options)) => {
          val modelOptions = if (options.isDefined) {
            TrainingOptions.fromArrayAndGeneralOptions(
              options.get,
              trainingOptions
            )
          } else {
            trainingOptions
          }
          val params = train(examples, modelOptions)
          Vector(id, params._1.toString, params._2).mkString("\t")
        }
      }
  }

  def trainWithSameOptions(
      sc: SparkContext,
      config: Config
  )(implicit c: ClassTag[T]): Unit = {
    val trainingConfig = TrainingConfig.loadConfig(config)
    val hc: HiveContext = new HiveContext(sc)

    val trainingExamples: RDD[(String, Seq[T])] =
      BinaryRegressionTrainer.getTraining(
        hc,
        trainingDataType,
        trainingConfig,
        DataLoadingRules.isEnoughSamplesToTrain
      )

    val trainingOptions: TrainingOptions =
      TrainingOptions
        .loadBaseTrainingOptions(
          config.getConfig("training_options"))

    val result = trainingExamples.
      map {
        case (id, examples) => {
          val params = train(examples, trainingOptions)
          Vector(id, params._1.toString, params._2).mkString("\t")
        }
      }
    HiveUtil.saveToHiveWithConfig(hc, config.getConfig("production"), result)
  }

  def trainWithPerModelOptions(
      sc: SparkContext,
      config: Config
  )(implicit c: ClassTag[T]): Unit = {
    val trainingConfig = TrainingConfig.loadConfig(config)
    val hc: HiveContext = new HiveContext(sc)

    val trainingExamples: RDD[(String, Seq[T])] =
      BinaryRegressionTrainer.getTraining(
        hc,
        trainingDataType,
        trainingConfig,
        DataLoadingRules.isEnoughSamplesToTrain
      )

    val trainingOptions: TrainingOptions =
      TrainingOptions
        .loadBaseTrainingOptions(
          config.getConfig("training_options"))

    val perModelOptions = HiveUtil.loadDataFromHive(
      hc,
      config.getString("options_query"),
      HiveUtil.parseLongToStringFromHiveRow("id_listing"),
      HiveUtil.parseDoubleArrayFromHiveRow("options")
    )

    val result = getParamsFromPerModelOptions(
      trainingExamples,
      perModelOptions,
      trainingOptions
    )

    HiveUtil.saveToHiveWithConfig(hc, config.getConfig("production"), result)
  }

  /**
    * Given a top-level configuration, perform a search training and evaluation job.
    * used in dev
    *
    * @param sc     SparkContext
    * @param config top-level config instance
    */
  def paramSearchPerModelWithEvalAndHoldout(
      sc: SparkContext,
      config: Config
  )(implicit c: ClassTag[T]): Unit = {
    // Pull out the search and eval configs for passing along
    val searchConf: BaseSearchConfig = BaseSearchConfig.loadConfig(config)
    val evalConfig: EvalConfig =
      DirectQueryEvalConfig.loadConfig(config)

    // Train using the given search and eval configs
    trainWithBaseSearchConfigPerModel(sc, searchConf, evalConfig)
  }

  /**
    * Given a top-level configuration, divide the training_data_query
    * into training/eval/holdout based on config.
    * used in production
    *
    * @param sc     SparkContext
    * @param config top-level config instance
    */
  def paramSearchPerModel(
      sc: SparkContext,
      config: Config
  )(implicit c: ClassTag[T]): Unit = {
    // Pull out the search and eval configs for passing along
    val searchConfig: BaseSearchConfig = BaseSearchConfig.loadConfig(config)
    val trainingConfig = TrainingConfig.loadConfig(config)

    val hc: HiveContext = new HiveContext(sc)
    val trainingOptionsArray: Array[TrainingOptions] = searchConfig.getTrainingOptions
    logger.info(s"Training models with ${trainingOptionsArray.mkString(",")}")
    val trainingExamples: RDD[(String, Seq[T])] =
      BinaryRegressionTrainer.getTraining(
        hc,
        trainingDataType,
        trainingConfig,
        DataLoadingRules.isEnoughSamplesToTrain
      )
    val results: RDD[ModelOutput[T]] =
      trainingExamples
        .map {
          case (id: String, data: Seq[T]) => {
            val samples = RandomUtil.sample(data, searchConfig.sampleRatio)
            // TODO cross validation
            searchBestOptions(id, samples.head, samples(1), samples(2), trainingOptionsArray)
          }
        }
        .cache()

    BinaryRegressionTrainer.saveAndEvalModelOutput(hc, results, searchConfig)

    // Clean up the results RDD
    results.unpersist()
  }

  /**
    * Using the config, load our data sets and use them to perform the training
    * search and evaluation.
    *
    * @param sc           SparkContext
    * @param searchConfig BaseSearchConfig instance with param search options
    * @param evalConfig   StrategyModelEvalConfig instance
    */
  def trainWithBaseSearchConfigPerModel(
      sc: SparkContext,
      searchConfig: BaseSearchConfig,
      evalConfig: EvalConfig
  )(implicit c: ClassTag[T]): Unit = {
    // Load training, eval, and holdout data feeds
    val hc: HiveContext = new HiveContext(sc)

    val trainingExamples: RDD[(String, Seq[T])] =
      BinaryRegressionTrainer.getTraining(
        hc,
        trainingDataType,
        evalConfig.trainingConfig,
        DataLoadingRules.isEnoughSamplesToTrain
      )

    logger.info("Loading eval data")
    val evalSamples: RDD[(String, Seq[T])] =
      trainingDataType.loadDataFromHive(hc, evalConfig.evalDataQuery)

    logger.info("Loading holdout data")
    val holdoutSamples: RDD[(String, Seq[T])] =
      trainingDataType.loadDataFromHive(hc, evalConfig.holdoutDataQuery)

    val trainingOptionsArray: Array[TrainingOptions] = searchConfig.getTrainingOptions
    logger.info(s"Training models with ${trainingOptionsArray.mkString(",")}")
    val results: RDD[ModelOutput[T]] =
      searchBestOptionsPerModel(
        trainingExamples,
        evalSamples,
        holdoutSamples,
        trainingOptionsArray
      ).cache()

    BinaryRegressionTrainer.saveAndEvalModelOutput(hc, results, searchConfig)

    // Clean up the results RDD
    results.unpersist()
  }

  /**
    * Given training, eval and holdout examples, join them together by modelId
    * and use them to grid search for the best training options.
    *
    * ** Note: This does not support initialization from existing parameters
    *
    * @param trainingExamples training data set
    * @param evalExamples     evaluation data set
    * @param holdoutExamples  holdout data set
    * @param optionArr        TrainingOptions to try
    * @return a ModelOutput for every model
    */
  def searchBestOptionsPerModel(
      trainingExamples: RDD[(String, Seq[T])],
      evalExamples: RDD[(String, Seq[T])],
      holdoutExamples: RDD[(String, Seq[T])],
      optionArr: Array[TrainingOptions]
  ): RDD[ModelOutput[T]] = {
    trainingExamples
      .join(evalExamples)
      .join(holdoutExamples)
      .map {
        case (id: String, ((training: Seq[T], eval: Seq[T]), holdout: Seq[T])) => {
          searchBestOptions(id, training, eval, holdout, optionArr)
        }
      }
  }

  /**
    * Grid search across our training options, perform training, and then
    * perform evaluation of the eval and holdout examples.
    *
    * @param id               modelId
    * @param trainingExamples training data set
    * @param evalExamples     eval data set
    * @param holdoutExamples  holdout dataset
    * @param optionArr        different TrainingOptions permutations to try
    * @return
    */
  def searchBestOptions(
      id: String,
      trainingExamples: Seq[T],
      evalExamples: Seq[T],
      holdoutExamples: Seq[T],
      optionArr: Array[TrainingOptions]
  ): ModelOutput[T] = {
    // Run the training and evaluation
    val ((params: StrategyParams[T], loss: Double), trainingOptions: TrainingOptions) =
      optionArr
        .map { option: TrainingOptions =>
          (trainAndEval(trainingExamples, evalExamples, option), option)
        }
        .min(Ordering.by {
          output: ((StrategyParams[T], Double), TrainingOptions) =>
            output._1._2
        })

    // Evaluate metrics for the the eval and holdout examples
    val evalMetrics: BinaryMetrics =
      BinaryRegressionTrainer.getMetrics(evalExamples, params)
    val holdoutMetrics: BinaryMetrics =
      BinaryRegressionTrainer.getMetrics(holdoutExamples, params)

    // Return a constructed ModelOutput instance
    ModelOutput(id, params, loss, evalMetrics, holdoutMetrics, trainingOptions)
  }

  /**
    * Given training and eval examples, perform a train and eval run.
    *
    * @param trainingExamples training data set
    * @param evalExamples     eval data set
    * @param defaultOptions   default training options for this model
    * @param inputParam       strategy params to start with
    * @param debug            true if debug mode is needed
    * @return (trainedParams, evalScore)
    */
  def trainAndEval(
      trainingExamples: Seq[T],
      evalExamples: Seq[T],
      defaultOptions: TrainingOptions,
      inputParam: Option[StrategyParams[T]] = None,
      debug: Boolean = false
  ): (StrategyParams[T], Double) = {
    val params: StrategyParams[T] =
      train(trainingExamples, defaultOptions, inputParam, debug)._1
    (params, eval(evalExamples, params, defaultOptions.evalRatio))
  }

  def eval(
      examples: Seq[T],
      params: StrategyParams[T],
      ratio: Double
  ): Double = {
    val evalResult: Seq[Double] =
      examples.map(example =>
        BinaryRegressionTrainer.evalExample(example, params, ratio)
      )
    evalResult.sum / evalResult.length
  }

  /**
    * Given a set of training examples and a collection of training options,
    * train strategy params using a bounded loss function.
    *
    * @param examples               training samples
    * @param defaultTrainingOptions training options
    * @param inputParam             initial strategy params
    * @param debug                  if true, print extra logging
    * @return trained strategy params
    */
  def train(
      examples: Seq[T],
      defaultTrainingOptions: TrainingOptions,
      inputParam: Option[StrategyParams[T]] = None,
      debug: Boolean = false
  ): (StrategyParams[T], Double) = {
    // Input:
    // - examples are corresponding to the same model
    // - each implementation defines its own params
    // Output:
    // - updated host rejection parameters
    var params: StrategyParams[T] =
      getOrDefaultParams(inputParam, defaultTrainingOptions)

    val sampleCount: Int = examples.length

    if (BinaryRegressionTrainer.enoughTrueSamples(defaultTrainingOptions, examples)) {
      // Decaying learning rates
      var r0: Double = defaultTrainingOptions.r0
      var r1: Double = defaultTrainingOptions.r1

      // Keep track of the lowest loss sum we've seen
      var minLossSum: Double = Double.MaxValue
      // Keep track of the best params we've encountered
      var bestParams: StrategyParams[T] = params

      // Break training up into mini-batches
      val numBatches: Int =
        Math.floor(sampleCount / defaultTrainingOptions.miniBatchSize).toInt
      val numEpochs = BinaryRegressionTrainer.adjustNumEpochs(defaultTrainingOptions, numBatches)
      // Train across many epoches
      for (i <- 0 until numEpochs) {
        // sample examples for training
        val shuffledSamples: Seq[T] = Random.shuffle(examples)

        // Compute the total loss across all of the batches
        var lossSum: Double = 0.0
        for (j <- 0 until numBatches) {
          val samples: Seq[T] =
            shuffledSamples.slice(
              j * defaultTrainingOptions.miniBatchSize,
              (j + 1) * defaultTrainingOptions.miniBatchSize)
          val (loss: Double, newParams: StrategyParams[T]) =
            trainMiniBatch(
              params,
              samples,
              defaultTrainingOptions,
              r0,
              r1)
          lossSum += loss
          params = newParams
        }

        // If the current strategy params are the best-performing
        // so far, keep them.
        if (lossSum < minLossSum) {
          minLossSum = lossSum
          bestParams = params
        }

        // If we're in debug mode, print out epoch info
        if (debug) {
          val avgLossSum: Double = lossSum / sampleCount.toDouble
          logger.info(
            s"Epoch $i, lossSum=$lossSum, avgLossSum=$avgLossSum" + bestParams.prettyPrint
          )
        }

        // Decay our learning rates after each epoch
        r0 = r0 * defaultTrainingOptions.rateDecay
        r1 = r1 * defaultTrainingOptions.rateDecay
      }

      // Ensure that our final loss is acceptable. If it is, return the
      // trained params.
      // If it is not, return the final trained strategy params.
      val minAvgLossSum: Double = minLossSum / sampleCount.toDouble
      if (BinaryRegressionTrainer.acceptLoss(minAvgLossSum, defaultTrainingOptions) &&
        bestParams.hasValidValue) {
        (bestParams, minAvgLossSum)
      } else {
        (params, minAvgLossSum)
      }
    } else {
      (params, BinaryRegressionTrainer.DefaultParamLoss)
    }
  }

  /**
    * Given training samples, initialize strategy params.
    * If `$initialize_from_global_prior` is set to false, load the initial
    * strategy params in from the param output of the prior DS.
    * If it is set to true (default) it will use the global prior, according to the StrategyParams
    *
    * @param hc              HiveContext
    * @param config          base config object
    * @param rawTrainingData (id, [TrainingData])
    * @param trainingOptions options for training
    * @return
    */
  def joinTrainingSamplesWithParams(
      hc: HiveContext,
      config: Config,
      rawTrainingData: RDD[(String, Seq[T])],
      trainingOptions: TrainingOptions
  )(implicit c: ClassTag[T]):
  RDD[(String, (Seq[T], Option[StrategyParams[T]]))] = {
    // Filter out training datas which don't have enough examples for an effective
    // training
    val trainingSamples: RDD[(String, Seq[T])] =
      rawTrainingData
        .filter(x => DataLoadingRules.isEnoughSamplesToTrain(x._2))

    // If initializeFromGlobalPrior is true, we initialize the model using global prior.
    // Otherwise, we initialize the model using previous day's output.
    val initializeFromGlobalPrior: Boolean =
      Try(config.getBoolean("initialize_from_global_prior")).getOrElse(true)

    if (initializeFromGlobalPrior) {
      val emptyStrategyParamInput: Option[StrategyParams[T]] = None
      trainingSamples
        .mapValues(sample => (sample, emptyStrategyParamInput))
    } else {
      // initialize using previous day's output
      val dsEval: String = config.getString("ds_eval")
      val paramsQuery: String = config.getString("params_query")
      val params: RDD[(String, StrategyParams[T])] =
        loadParamsFromHive(
          hc,
          // Substitute in a new value for $DS_EVAL to get the prior day's params
          paramsQuery.replace("$DS_EVAL", PipelineUtil.dateMinus(dsEval, 1)))

      // Join yesterday's params with today's training data
      trainingSamples.leftOuterJoin(params)
    }
  }

  /**
    * Given a batch of training samples, compute a round of training.
    * This includes computing the collective loss and param gradients
    * across all of the samples.
    *
    * @param params  strategy params to base scoring on
    * @param samples training samples in the mini-batch
    * @param options training options
    * @param r0      learning rate r0
    * @param r1      learning rate r1
    * @return
    */
  def trainMiniBatch(
      params: StrategyParams[T],
      samples: Seq[T],
      options: TrainingOptions,
      r0: Double,
      r1: Double
  ): (Double, StrategyParams[T]) = {
    var lossSum: Double = 0.0
    val gradSum: Array[Double] =
      Array.fill[Double](params.params.length)(0)

    // For each of the samples, compute loss and gradient
    samples
      .foreach(example => {
        val (loss: Double, grad: Double) =
          getLossAndGradient(example, params, options)
        val gradUpdate: Array[Double] =
          params.computeGradient(grad, example)
        // Update our gradient array
        updateGradientSum(
          r0,
          r1,
          options,
          example,
          gradUpdate,
          gradSum)

        // Keep a loss total
        lossSum += loss
      })

    (lossSum, params.updatedParams(gradSum, options))
  }

  /**
    * Given current gradients and an array of update gradients,
    * update the gradients in-place on the `gradients` array.
    *
    * @param r0              learning rate r0
    * @param r1              learning rate r1
    * @param options         training options
    * @param example         the sample to
    * @param updateGradients gradient deltas for updating `gradients`
    * @param gradients       mutable gradients array to update in-place
    */
  def updateGradientSum(
      r0: Double,
      r1: Double,
      options: TrainingOptions,
      example: T,
      updateGradients: Array[Double],
      gradients: Array[Double]
  ): Unit = {
    // Ensure that our existing gradients and our updates have
    // the same number of params
    assert(
      updateGradients.length == gradients.length,
      s"${updateGradients.length}, ${gradients.length}")

    // Update each of the param gradients using the update array
    if (BinaryRegressionTrainer.validGradient(gradients)) {
      val learningRate: Double =
        getLearningRate(r0, r1, example, options)
      for (i <- gradients.indices) {
        gradients(i) += learningRate * updateGradients(i)
      }
    }
  }

  /**
    * Given a training sample, score it and get the loss and gradient
    * for it, given the passed-in model params.
    *
    * @param example training sample
    * @param params  current strategy params
    * @param options training options
    * @return (loss, gradient)
    */
  def getLossAndGradient(
      example: T,
      params: StrategyParams[T],
      options: TrainingOptions
  ): (Double, Double) = {
    // get loss and the gradient of loss w.r.t to model score
    val score: Double = params.score(example)
    var absLoss: Double = 0.0
    var grad: Double = 0.0

    // optimal value should lie in [lowerBound, upperBound]
    val lowerBound: Double = example.getLowerBound(options)
    lazy val upperBound: Double = example.getUpperBound(options)
    if (score < lowerBound) {
      absLoss += lowerBound - score
      grad -= 1.0
    } else if (score > upperBound) {
      absLoss += score - upperBound
      grad += 1.0
    } else {
      // no loss
    }

    // if `maxAvgLossRatio` is set and valid, compute loss as
    // a ratio of the `pivotValue`.
    if (options.maxAvgLossRatio > 0) {
      (absLoss / example.scoringPivot, grad)
    } else {
      (absLoss, grad)
    }
  }

  /**
    * Get params from an Option[StrategyParams], falling back to the
    * default specified in training options if not present.
    *
    * @param paramsOpt       strategy params
    * @param trainingOptions training options with a default
    * @return
    */
  protected def getOrDefaultParams(
      paramsOpt: Option[StrategyParams[T]],
      trainingOptions: TrainingOptions
  ): StrategyParams[T] = {
    if (paramsOpt.isDefined) {
      paramsOpt.get
    } else {
      getDefaultParams(trainingOptions)
    }
  }
}

object BinaryRegressionTrainer extends ScalaLogging {

  // TODO move it to config
  private final val MaxAvgLossSum      : Int    = 1000
  private final val DefaultParamLoss   : Double = -1
  private final val MinEpochs          : Int    = 10
  private final val MAXEpochSampleCount: Int    = 2000

  /**
    * Given StrategyParams, evaluate a sample
    *
    * @param example sample to evaluate
    * @param params  trained params to use
    * @param ratio   unused
    * @return
    */
  def evalExample[T <: BinaryTrainingSample](
      example: T,
      params: StrategyParams[T],
      // ratio indicates relative weights assigned to positive labels.
      ratio: Double
  ): Double = {
    val prediction: Double = params.score(example)
    example.lossRatioWithPrediction(prediction, ratio)
  }

  /**
    * Given params and an evaluation data set, compute our performance metrics.
    *
    * @param evalExamples evaluation data set
    * @param params       trained strategy params
    * @return a populated BinaryMetrics instance
    */
  def getMetrics[T <: BinaryTrainingSample](
      evalExamples: Seq[T],
      params: StrategyParams[T]
  ): BinaryMetrics = {
    val byKey: Seq[((Boolean, Boolean), Double)] =
      evalExamples
        .map { example: T =>
          val prediction: Double = params.score(example)
          val predictionLower: Boolean = example.predictionLower(prediction)
          val predictionIncrease: Double = example.predictionIncrease(prediction)
          (example.label, predictionLower) -> predictionIncrease
        }

    val results: Map[(Boolean, Boolean), (Int, Double)] =
      byKey
        // Group by key
        .groupBy(_._1)
        .map {
          case (key: (Boolean, Boolean), predictions: Seq[((Boolean, Boolean), Double)]) => {
            val count: Int = predictions.length
            val sum: Double = predictions.map(_._2).sum
            (key, (count, sum))
          }
        }

    // Get the regret cost at the 50th and 75th percentiles.
    trueRegret(byKey, List(0.50, 0.75)) match {
      case Seq(regret50: Double, regret75: Double) =>
        // Use them to compute our eval metrics
      {
        BinaryMetrics.computeEvalMetricFromCounts(results, regret50, regret75)
      }
    }
  }

  /**
    * Calculate true regret of the prediction
    *
    * @param predictions          evaluation predictions to find regret from
    * @param percentilesToCompute ie, [0.25, 0.5, 0.75]
    * @return the true regret at the requested percentiles
    */
  def trueRegret(
      // ((label, predictionLower), predictionIncrease)
      predictions: Seq[((Boolean, Boolean), Double)],
      percentilesToCompute: List[Double]
  ): List[Double] = {
    // Filter out examples where label is true but prediction was lower
    val regret: Seq[Double] =
      predictions
        .filter {
          case ((label: Boolean, predictionLower: Boolean), _) => {
            label && predictionLower
          }
        }
        .map(_._2)

    // If we have examples remaining, retrieve their regret at the requested percentiles
    if (regret.nonEmpty) {
      percentilesToCompute
        .map { percentile: Double =>
          val pos: Int = (percentile * regret.length.toDouble).floor.toInt
          Sort.quickSelect(regret, pos)
        }
    } else {
      List.fill(percentilesToCompute.length)(0)
    }
  }

  def enoughTrueSamples[T <: BinaryTrainingSample](options: TrainingOptions, examples: Seq[T]):
  Boolean = {
    val count = options.minTrueLabelCount
    if (count == 0) {
      true
    } else {
      examples.count(_.label) >= count
    }
  }

  private def validGradient(gradients: Array[Double]): Boolean = {
    gradients.forall(!_.isNaN)
  }

  /**
    * Is this loss acceptable?
    * If we've defined a valid maxAvgLossRatio, we compare it as a per-sample
    * ratio. If we haven't defined it, it's compared to a total count.
    *
    * @param loss   loss ratio or loss sum to check
    * @param option training options
    * @return true if the loss is within acceptable bounds
    */
  private def acceptLoss(loss: Double, option: TrainingOptions): Boolean = {
    if (option.maxAvgLossRatio > 0) {
      loss <= option.maxAvgLossRatio
    } else {
      loss <= BinaryRegressionTrainer.MaxAvgLossSum
    }
  }

  def getTraining[T <: BinaryTrainingSample](
      hc: HiveContext,
      trainingData: TrainingData[T],
      config: TrainingConfig,
      isEnoughSamplesToTrain: Seq[T] => Boolean
  )(implicit c: ClassTag[T]): RDD[(String, Seq[T])] = {
    logger.info("Loading training data")
    trainingData.loadDataFromHive(hc, config.trainingDataQuery)
      .filter(x => isEnoughSamplesToTrain(x._2))
      // Shuffle distributes data evenly and with each partition has a lot less data
      // executor is less likely to spill. If you have 500+ executor cores, it is always
      // good to shuffle. Otherwise the cost of shuffling might be big enough and the lack
      // of parallelism makes it not worth it.
      //
      // NB: If uniform sampling is performed via simple mod, caution should be exercised here
      // to make sure the hashcode can be uniformed distributed.
      // For example, the current implementation is a String based hashcode which would not
      // correlate
      // with simple mod on id_listing.
      // However, if we switch to Long based key, then the repartition number should be co-prime
      // with modular used in sampling.
      .coalesce(config.partitions, config.shuffle)
  }

  def saveAndEvalModelOutput[T <: BinaryTrainingSample](
      hc: HiveContext,
      result: RDD[ModelOutput[T]],
      searchConfig: BaseSearchConfig
  ): Unit = {
    // Ensure that the incoming RDD is persisted, otherwise this method will
    // force recomputations and have poor performance.
    assert(
      result.getStorageLevel != StorageLevel.NONE,
      "`result` RDD must be persisted outside of this method or it may force re-training."
    )

    // Save the training results
    ModelOutput.save(hc, result, searchConfig.table, searchConfig.partition)

    // Log evaluation metrics
    logger.info(BinaryMetrics.metricsHeader)
    logger.info(
      "evalMetrics: " +
        BinaryMetrics.combineEvalMetricFromRDD(result.map(_.evalMetrics)).toString)
    logger.info(
      "holdoutMetrics: " +
        BinaryMetrics.combineEvalMetricFromRDD(result.map(_.holdoutMetrics)).toString)

    // TODO make it configurable
    BinaryRegressionTrainer.computeStats(result)
  }

  private def computeStats[T <: BinaryTrainingSample](result: RDD[ModelOutput[T]]) = {
    val statsData = result.map { x => {
      (Vectors.dense(x.params.params), Vectors.dense(x.options.toArray.take(4)))
    }
    }
    val paramStats = Statistics.colStats(statsData.map(x => x._1))
    val paramColmean = paramStats.mean.toArray.mkString(",")
    val paramColVariance = paramStats.variance.toArray.mkString(",")
    logger.info("paramsMean: " + paramColmean)
    logger.info("paramsVariance: " + paramColVariance)

    // we only get stats for hyper params: trueLowerBound, falseUpperBound, falseLowerBound, trueUpperBound
    val optionStats = Statistics.colStats(statsData.map(x => x._2))
    val optionColmean = optionStats.mean.toArray.mkString(",")
    val optionColVariance = optionStats.variance.toArray.mkString(",")
    logger.info("optionsMean: " + optionColmean)
    logger.info("optionsVariance: " + optionColVariance)
  }

  def adjustNumEpochs(defaultTrainingOptions: TrainingOptions, numBatches: Int): Int = {
    val maxBatchesCount = MAXEpochSampleCount /
      defaultTrainingOptions.miniBatchSize
    if (numBatches > maxBatchesCount) {
      val d = numBatches / maxBatchesCount
      math.max(MinEpochs, defaultTrainingOptions.numEpochs / d)
    } else {
      defaultTrainingOptions.numEpochs
    }
  }
}
