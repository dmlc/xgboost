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

import ml.dmlc.xgboost4j.scala.spark.TrackerConf

import org.apache.spark.ml.param._

trait GeneralParams extends Params {

  /**
   * The number of rounds for boosting
   */
  val round = new IntParam(this, "num_round", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  /**
   * number of workers used to train xgboost model. default: 1
   */
  val nWorkers = new IntParam(this, "nworkers", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))

  /**
   * number of threads used by per worker. default 1
   */
  val numThreadPerTask = new IntParam(this, "nthread", "number of threads used by per worker",
    ParamValidators.gtEq(1))

  /**
   * whether to use external memory as cache. default: false
   */
  val useExternalMemory = new BooleanParam(this, "use_external_memory", "whether to use external" +
    "memory as cache")

  /**
   * 0 means printing running messages, 1 means silent mode. default: 0
   */
  val silent = new IntParam(this, "silent",
    "0 means printing running messages, 1 means silent mode.",
    (value: Int) => value >= 0 && value <= 1)

  /**
   * customized objective function provided by user. default: null
   */
  val customObj = new CustomObjParam(this, "custom_obj", "customized objective function " +
    "provided by user")

  /**
   * customized evaluation function provided by user. default: null
   */
  val customEval = new CustomEvalParam(this, "custom_eval", "customized evaluation function " +
    "provided by user")

  /**
   * the value treated as missing. default: Float.NaN
   */
  val missing = new FloatParam(this, "missing", "the value treated as missing")

  /**
    * the interval to check whether total numCores is no smaller than nWorkers. default: 30 minutes
    */
  val timeoutRequestWorkers = new LongParam(this, "timeout_request_workers", "the maximum time to" +
    " request new Workers if numCores are insufficient. The timeout will be disabled if this" +
    " value is set smaller than or equal to 0.")

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
  val trackerConf = new TrackerConfParam(this, "tracker_conf", "Rabit tracker configurations")

  /** Random seed for the C++ part of XGBoost and train/test splitting. */
  val seed = new LongParam(this, "seed", "random seed")

  setDefault(round -> 1, nWorkers -> 1, numThreadPerTask -> 1,
    useExternalMemory -> false, silent -> 0,
    customObj -> null, customEval -> null, missing -> Float.NaN,
    trackerConf -> TrackerConf(), seed -> 0, timeoutRequestWorkers -> 30 * 60 * 1000L
  )
}
