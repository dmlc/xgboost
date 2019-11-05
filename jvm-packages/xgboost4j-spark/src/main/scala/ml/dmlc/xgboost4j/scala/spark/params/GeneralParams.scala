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

package ml.dmlc.xgboost4j.scala.spark.params

import com.google.common.base.CaseFormat
import ml.dmlc.xgboost4j.scala.spark.TrackerConf

import org.apache.spark.ml.param._
import scala.collection.mutable

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{FloatType, IntegerType}

private[spark] trait GeneralParams extends Params {

  /**
   * The number of rounds for boosting
   */
  final val numRound = new IntParam(this, "numRound", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  final def getNumRound: Int = $(numRound)

  /**
   * number of workers used to train xgboost model. default: 1
   */
  final val numWorkers = new IntParam(this, "numWorkers", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))

  final def getNumWorkers: Int = $(numWorkers)

  /**
   * number of threads used by per worker. default 1
   */
  final val nthread = new IntParam(this, "nthread", "number of threads used by per worker",
    ParamValidators.gtEq(1))

  final def getNthread: Int = $(nthread)

  /**
   * whether to use external memory as cache. default: false
   */
  final val useExternalMemory = new BooleanParam(this, "useExternalMemory",
    "whether to use external memory as cache")

  final def getUseExternalMemory: Boolean = $(useExternalMemory)

  /**
   * Deprecated. Please use verbosity instead.
   * 0 means printing running messages, 1 means silent mode. default: 0
   */
  final val silent = new IntParam(this, "silent",
    "Deprecated. Please use verbosity instead. " +
    "0 means printing running messages, 1 means silent mode.",
    (value: Int) => value >= 0 && value <= 1)

  final def getSilent: Int = $(silent)

  /**
   * Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
   * default: 1
   */
  final val verbosity = new IntParam(this, "verbosity",
    "Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), " +
    "3 (debug).",
    (value: Int) => value >= 0 && value <= 3)

  final def getVerbosity: Int = $(verbosity)

  /**
   * customized objective function provided by user. default: null
   */
  final val customObj = new CustomObjParam(this, "customObj", "customized objective function " +
    "provided by user")

  /**
   * customized evaluation function provided by user. default: null
   */
  final val customEval = new CustomEvalParam(this, "customEval",
    "customized evaluation function provided by user")

  /**
   * the value treated as missing. default: Float.NaN
   */
  final val missing = new FloatParam(this, "missing", "the value treated as missing")

  final def getMissing: Float = $(missing)

  /**
    * the maximum time to wait for the job requesting new workers. default: 30 minutes
    */
  final val timeoutRequestWorkers = new LongParam(this, "timeoutRequestWorkers", "the maximum " +
    "time to request new Workers if numCores are insufficient. The timeout will be disabled " +
    "if this value is set smaller than or equal to 0.")

  final def getTimeoutRequestWorkers: Long = $(timeoutRequestWorkers)

  /**
    * The hdfs folder to load and save checkpoint boosters. default: `empty_string`
    */
  final val checkpointPath = new Param[String](this, "checkpointPath", "the hdfs folder to load " +
    "and save checkpoints. If there are existing checkpoints in checkpoint_path. The job will " +
    "load the checkpoint with highest version as the starting point for training. If " +
    "checkpoint_interval is also set, the job will save a checkpoint every a few rounds.")

  final def getCheckpointPath: String = $(checkpointPath)

  /**
    * Param for set checkpoint interval (&gt;= 1) or disable checkpoint (-1). E.g. 10 means that
    * the trained model will get checkpointed every 10 iterations. Note: `checkpoint_path` must
    * also be set if the checkpoint interval is greater than 0.
    */
  final val checkpointInterval: IntParam = new IntParam(this, "checkpointInterval",
    "set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the trained " +
      "model will get checkpointed every 10 iterations. Note: `checkpoint_path` must also be " +
      "set if the checkpoint interval is greater than 0.",
    (interval: Int) => interval == -1 || interval >= 1)

  final def getCheckpointInterval: Int = $(checkpointInterval)

  /**
    * Rabit tracker configurations. The parameter must be provided as an instance of the
    * TrackerConf class, which has the following definition:
    *
    *     case class TrackerConf(workerConnectionTimeout: Duration, trainingTimeout: Duration,
    *                            trackerImpl: String)
    *
    * See below for detailed explanations.
    *
    *   - trackerImpl: Select the implementation of Rabit tracker.
    *                  default: "python"
    *
    *        Choice between "python" or "scala". The former utilizes the Java wrapper of the
    *        Python Rabit tracker (in dmlc_core), and does not support timeout settings.
    *        The "scala" version removes Python components, and fully supports timeout settings.
    *
    *   - workerConnectionTimeout: the maximum wait time for all workers to connect to the tracker.
    *                             default: 0 millisecond (no timeout)
    *
    *        The timeout value should take the time of data loading and pre-processing into account,
    *        due to the lazy execution of Spark's operations. Alternatively, you may force Spark to
    *        perform data transformation before calling XGBoost.train(), so that this timeout truly
    *        reflects the connection delay. Set a reasonable timeout value to prevent model
    *        training/testing from hanging indefinitely, possible due to network issues.
    *        Note that zero timeout value means to wait indefinitely (equivalent to Duration.Inf).
    *        Ignored if the tracker implementation is "python".
    */
  final val trackerConf = new TrackerConfParam(this, "trackerConf", "Rabit tracker configurations")

  /** Random seed for the C++ part of XGBoost and train/test splitting. */
  final val seed = new LongParam(this, "seed", "random seed")

  final def getSeed: Long = $(seed)

  setDefault(numRound -> 1, numWorkers -> 1, nthread -> 1,
    useExternalMemory -> false, silent -> 0, verbosity -> 1,
    customObj -> null, customEval -> null, missing -> Float.NaN,
    trackerConf -> TrackerConf(), seed -> 0, timeoutRequestWorkers -> 30 * 60 * 1000L,
    checkpointPath -> "", checkpointInterval -> -1)
}

trait HasLeafPredictionCol extends Params {
  /**
   * Param for leaf prediction column name.
   * @group param
   */
  final val leafPredictionCol: Param[String] = new Param[String](this, "leafPredictionCol",
    "name of the predictLeaf results")

  /** @group getParam */
  final def getLeafPredictionCol: String = $(leafPredictionCol)
}

trait HasContribPredictionCol extends Params {
  /**
   * Param for contribution prediction column name.
   * @group param
   */
  final val contribPredictionCol: Param[String] = new Param[String](this, "contribPredictionCol",
    "name of the predictContrib results")

  /** @group getParam */
  final def getContribPredictionCol: String = $(contribPredictionCol)
}

trait HasBaseMarginCol extends Params {

  /**
   * Param for initial prediction (aka base margin) column name.
   * @group param
   */
  final val baseMarginCol: Param[String] = new Param[String](this, "baseMarginCol",
    "Initial prediction (aka base margin) column name.")

  /** @group getParam */
  final def getBaseMarginCol: String = $(baseMarginCol)
}

trait HasGroupCol extends Params {

  /**
   * Param for group column name.
   * @group param
   */
  final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name.")

  /** @group getParam */
  final def getGroupCol: String = $(groupCol)

}

trait HasNumClass extends Params {

  /**
   * number of classes
   */
  final val numClass = new IntParam(this, "numClass", "number of classes")

  /** @group getParam */
  final def getNumClass: Int = $(numClass)
}

private[spark] trait ParamMapFuncs extends Params {

  def XGBoost2MLlibParams(xgboostParams: Map[String, Any]): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      if ((paramName == "booster" && paramValue != "gbtree") ||
        (paramName == "updater" && paramValue != "grow_histmaker,prune" &&
          paramValue != "hist")) {
        throw new IllegalArgumentException(s"you specified $paramName as $paramValue," +
          s" XGBoost-Spark only supports gbtree as booster type" +
          " and grow_histmaker,prune or hist as the updater type")
      }
      val name = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, paramName)
      params.find(_.name == name).foreach {
        case _: DoubleParam =>
          set(name, paramValue.toString.toDouble)
        case _: BooleanParam =>
          set(name, paramValue.toString.toBoolean)
        case _: IntParam =>
          set(name, paramValue.toString.toInt)
        case _: FloatParam =>
          set(name, paramValue.toString.toFloat)
        case _: LongParam =>
          set(name, paramValue.toString.toLong)
        case _: Param[_] =>
          set(name, paramValue)
      }
    }
  }

  def MLlib2XGBoostParams: Map[String, Any] = {
    val xgboostParams = new mutable.HashMap[String, Any]()
    for (param <- params) {
      if (isDefined(param)) {
        val name = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, param.name)
        xgboostParams += name -> $(param)
      }
    }
    xgboostParams.toMap
  }
}
