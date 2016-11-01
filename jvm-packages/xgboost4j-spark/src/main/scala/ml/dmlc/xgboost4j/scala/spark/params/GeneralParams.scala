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

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.param._

import scala.concurrent.duration.{SECONDS, Duration}

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
  val customObj = new Param[ObjectiveTrait](this, "custom_obj", "customized objective function " +
    "provided by user")

  /**
   * customized evaluation function provided by user. default: null
   */
  val customEval = new Param[EvalTrait](this, "custom_obj", "customized evaluation function " +
    "provided by user")

  /**
   * the value treated as missing. default: Float.NaN
   */
  val missing = new FloatParam(this, "missing", "the value treated as missing")

  /**
   * select the implementation of Rabit tracker. default: "python"
   * choice between "python" or "scala". The former utilizes the Python-version
   * of Rabit tracker (in dmlc_core), whereas the latter is implemented in Akka
   * without Python components, suitable for users experiencing issues with
   * Python. The Akka implementation is currently experimental, use at your own risk.
   */
  val trackerImpl = new Param[String](this, "tracker_impl",
    "implementation of the Rabit tracker, choice between \"python\" and \"scala\"")

  /**
   * the maximum wait time for all workers to connect to the tracker.
   * the timeout value should take the time of data loading and preprocessing into account, due
   * to the lazy execution of Spark's operations.
   * set a reasonable timeout value to prevent model training/testing from hanging indefinitely.
   * ignored if the tracker implementation is "python".
   *
   * default: Long.MaxValue seconds
   */
  val workerConnectionTimeout = new Param[Duration](this, "worker_connection_timeout",
    "the timeout for all workers to connect to the tracker. " +
    "Use a finite timeout duration to prevent tracker from hanging indefinitely."
  )

  /**
   * the maximum time for training a model with xgboost4j-spark.
   * the training will hang indefinitely if an executor is lost during the training process,
   * and setting a finite timeout value prevents hanging.
   * ignored if the tracker implementation is "python".
   *
   * default: Long.MaxValue seconds
   */
  val trainingTimeout = new Param[Duration](this, "training_timeout",
    "the timeout for model-training.")

  setDefault(round -> 1, nWorkers -> 1, numThreadPerTask -> 1,
    useExternalMemory -> false, silent -> 0,
    customObj -> null, customEval -> null, missing -> Float.NaN,
    trackerImpl -> "python",
    workerConnectionTimeout -> Duration.apply(Long.MaxValue, SECONDS),
    trainingTimeout -> Duration.apply(Long.MaxValue, SECONDS)
  )
}
