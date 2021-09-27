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

package ml.dmlc.xgboost4j.scala.spark.rapids

import java.util.concurrent.{LinkedBlockingQueue, ThreadPoolExecutor, TimeUnit, TimeoutException}

import com.google.common.util.concurrent.{MoreExecutors, ThreadFactoryBuilder}
import ml.dmlc.xgboost4j.scala.spark.rapids.JasonWrapper._
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.tuning.{CrossValidatorModel, RapidsCrossValidatorModel}
import org.apache.spark.ml.{Model, PipelineStage, tuning}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.json4s.JsonDSL.{map2jvalue, _}
import org.json4s.{JValue, _}
import org.json4s.jackson.JsonMethods.{compact, render}

import scala.concurrent.duration.Duration
import scala.concurrent.{Awaitable, ExecutionContext, Future}
import scala.util.control.NonFatal

/*
 * @param lb lower bound of the acceptance range
 * @param ub upper bound of the acceptance range
 * @param seed
 * @param complement whether to use the complement of the range specified, default to false
 */
case class GpuSampler(lb: Double, ub: Double, seed: Long, complement: Boolean = false) {
  val roundingEpsilon = 1e-6
  /** epsilon slop to avoid failure from floating point jitter. */
  require(
    lb <= (ub + roundingEpsilon),
    s"Lower bound ($lb) must be <= upper bound ($ub)")
  require(
    lb >= (0.0 - roundingEpsilon),
    s"Lower bound ($lb) must be >= 0.0")
  require(
    ub <= (1.0 + roundingEpsilon),
    s"Upper bound ($ub) must be <= 1.0")
}

class CrossValidator extends tuning.CrossValidator {

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    if (GpuUtils.isRapidsEnabled(dataset)) {
      train(dataset)
    } else {
      super.fit(dataset)
    }
  }

  /**
   * Fits a single model to the input data.
   *
   * @param dataset input Dataset
   * @return best Model which should be used as asInstanceOf[XGBoostClassificationModel] or
   *         asInstanceOf[XGBoostRegressionModel] accordingly
   */
  private def train(dataset: Dataset[_]): CrossValidatorModel = {
    val instr = new Instrumentation
    val schema = dataset.schema
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContexts

    instr.logPipelineStage(this)
    instr.logParams(this, numFolds, seed, parallelism)
    logTuningParam(instr)

    val collectSubModelsParam = $(collectSubModels)

    var subModels: Option[Array[Array[Model[_]]]] = if (collectSubModelsParam) {
      Some(Array.fill($(numFolds))(Array.fill[Model[_]](epm.length)(null)))
    } else None

    val metrics = (0 until $(numFolds)).map(splitIndex => {
      val lb = splitIndex / $(numFolds).toFloat
      val up = (splitIndex + 1) / $(numFolds).toFloat

      // Fit models in a Future for training in parallel
      val foldMetricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {

          val trainingSampler = Some(GpuSampler(lb, up, $(seed), true))
          val validationSampler = Some(GpuSampler(lb, up, $(seed), false))

          instr.logDebug(s"Train split $splitIndex with multiple sets of parameters.")
          var model: Model[_] = null

          val evalDf = est match {
            case classifier: XGBoostClassifier =>
              model = classifier.copy(paramMap).trainWithGpuSampler(dataset, trainingSampler)
              model match {
                case classifierModel: XGBoostClassificationModel =>
                  classifierModel
                    .set(classifierModel.toRowCols, Seq(classifier.getLabelCol))
                    .transformWithGpuSampler(dataset, validationSampler)
              }
            case regressor: XGBoostRegressor =>
              model = regressor.copy(paramMap).trainWithGpuSampler(dataset, trainingSampler)
              model match {
                case regressorModel: XGBoostRegressionModel =>
                  regressorModel
                    .set(regressorModel.toRowCols, Seq(regressor.getLabelCol))
                    .transformWithGpuSampler(dataset, validationSampler)
              }
            case _ => throw new IllegalArgumentException("Only XGBoostRegressor and " +
              "XGBoostClassifier are supported"
            )
          }

          if (collectSubModelsParam) {
            subModels.get(splitIndex)(paramIndex) = model
          }

          val metric = eval.evaluate(evalDf)
          instr.logDebug(s"Got metric $metric for model trained with $paramMap.")
          metric
        }(executionContext)
      }

      // Wait for metrics to be calculated
      val foldMetrics = foldMetricFutures.map(awaitResult(_, Duration.Inf))
      foldMetrics
    }).transpose.map(_.sum / $(numFolds)) // Calculate average metric over all splits

    instr.logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)

    instr.logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    instr.logInfo(s"Best cross-validation metric: $bestMetric.")

    val bestModel = est match {
      case classifier: XGBoostClassifier =>
        classifier.copy(epm(bestIndex)).fit(dataset).asInstanceOf[Model[_]]
      case regressor: XGBoostRegressor =>
        regressor.copy(epm(bestIndex)).fit(dataset).asInstanceOf[Model[_]]
      case _ => throw new IllegalArgumentException("Only XGBoostRegressor and " +
        "XGBoostClassifier are supported"
      )
    }

    copyValues(RapidsCrossValidatorModel.createCrossValidatorModel(
      uid, bestModel, metrics.toArray, subModels).setParent(this))
  }

  private def awaitResult[T](awaitable: Awaitable[T], atMost: Duration): T = {
    try {
      // `awaitPermission` is not actually used anywhere so it's safe to pass in null here.
      // See SPARK-13747.
      val awaitPermission = null.asInstanceOf[scala.concurrent.CanAwait]
      awaitable.result(atMost)(awaitPermission)
    } catch {
      //      case e: SparkFatalException =>
      //        throw e.throwable
      // TimeoutException is thrown in the current thread, so not need to warp the exception.
      case NonFatal(t) if !t.isInstanceOf[TimeoutException] =>
        throw new SparkException("Exception thrown in awaitResult: ", t)
    }
  }

  private def newDaemonCachedThreadPool(
      prefix: String, maxThreadNumber: Int, keepAliveSeconds: Int = 60): ThreadPoolExecutor = {
    val threadFactory = new ThreadFactoryBuilder().setDaemon(true).setNameFormat(prefix + "-%d")
      .build()
    val threadPool = new ThreadPoolExecutor(
      maxThreadNumber, // corePoolSize: the max number of threads to create before queuing the tasks
      maxThreadNumber, // maximumPoolSize: because we use LinkedBlockingDeque, this one is not used
      keepAliveSeconds,
      TimeUnit.SECONDS,
      new LinkedBlockingQueue[Runnable],
      threadFactory)
    threadPool.allowCoreThreadTimeOut(true)
    threadPool
  }

  private def getExecutionContexts: ExecutionContext = {
    getParallelism match {
      case 1 =>
        ExecutionContext.fromExecutorService(MoreExecutors.sameThreadExecutor())
      case n =>
        ExecutionContext.fromExecutorService(
          newDaemonCachedThreadPool(s"${
            this.getClass.getSimpleName
          }-thread-pool", n))
    }
  }

  private def logParams(hasParams: Params, params: Param[_]*): Unit = {
    val pairs: Seq[(String, JValue)] = for {
      p <- params
      value <- hasParams.get(p)
    } yield {
      val cast = p.asInstanceOf[Param[Any]]
      p.name -> parse(cast.jsonEncode(value))
    }
    logInfo(compact(render(map2jvalue(pairs.toMap))))
  }

  private def logTuningParam(instrumentation: Instrumentation): Unit = {
    instrumentation.logNamedValue("estimator", $(estimator).getClass.getCanonicalName)
    instrumentation.logNamedValue("evaluator", $(evaluator).getClass.getCanonicalName)
    instrumentation.logNamedValue("estimatorParamMapsLength", $(estimatorParamMaps).length)
  }
}

private class Instrumentation extends Logging {
  val prefix = "GPU CrossValidator "

  override def logInfo(msg: => String): Unit = {
    super.logInfo(prefix + msg)
  }

  /**
   * Logs a debug message with a prefix that uniquely identifies the training session.
   */
  override def logDebug(msg: => String): Unit = {
    super.logDebug(prefix + msg)
  }

  def logParams(hasParams: Params, params: Param[_]*): Unit = {
    val pairs: Seq[(String, JValue)] = for {
      p <- params
      value <- hasParams.get(p)
    } yield {
      val cast = p.asInstanceOf[Param[Any]]
      p.name -> parse(cast.jsonEncode(value))
    }
    logInfo(compact(render(map2jvalue(pairs.toMap))))
  }

  /**
   * Log some data about the dataset being fit.
   */
  def logDataset(dataset: Dataset[_]): Unit = logDataset(dataset.rdd)

  /**
   * Log some data about the dataset being fit.
   */
  def logDataset(dataset: RDD[_]): Unit = {
    logInfo(s"training: numPartitions=${dataset.partitions.length}" +
      s" storageLevel=${dataset.getStorageLevel}")
  }

  /**
   * Logs the value with customized name field.
   */
  def logNamedValue(name: String, value: String): Unit = {
    logInfo(compact(render(name -> value)))
  }

  def logNamedValue(name: String, value: Long): Unit = {
    logInfo(compact(render(name -> value)))
  }

  /**
   * Log some info about the pipeline stage being fit.
   */
  def logPipelineStage(stage: PipelineStage): Unit = {
    // estimator.getClass.getSimpleName can cause Malformed class name error,
    // call safer `Utils.getSimpleName` instead
    val className = getSimpleName(stage.getClass)
    logInfo(s"Stage class: $className")
    logInfo(s"Stage uid: ${stage.uid}")
  }

  def getSimpleName(cls: Class[_]): String = {
    try {
      cls.getSimpleName
    } catch {
      case err: InternalError => "Unknow"
    }
  }
}
