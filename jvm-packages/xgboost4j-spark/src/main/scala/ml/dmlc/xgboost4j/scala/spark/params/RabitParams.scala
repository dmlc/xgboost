/*
 Copyright (c) 2014 - 2019 by Contributors

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

import org.apache.spark.ml.param._

private[spark] trait RabitParams extends Params {
  /**
   * Rabit parameters passed through Rabit.Init into native layer
   * rabit_ring_reduce - enable ring based allreduce/broadcast operations.
   * rabit_reduce_buffer - buffer size to recv and run reduction
   * rabit_debug - enable more verbose rabit logging to stdout
   * rabit_timeout - wait interval before exit after rabit observed failures set -1 to disable
   * dmlc_worker_connect_retry - number of retrys to tracker
   * dmlc_worker_stop_process_on_error - exit process when rabit see assert/error
   */
  final val ringReduce = new BooleanParam(this, "rabitRingReduce",
    "enable allreduce/broadcast with ring based topology")

  final def reduceBuffer: Param[String] = new Param[String](this, "rabitReduceBuffer",
    "buffer size (MB/GB) allocated to each xgb trainner recv and run reduction",
    (buf: String) => buf.contains("MB") || buf.contains("GB"))

  final def debug: BooleanParam = new BooleanParam(this, "rabitDebug",
    "enable more verbose rabit logging to stdout")

  final def timeout: IntParam = new IntParam(this, "rabitTimeout",
  "timeout threshold after rabit observed failures")

  final def connectRetry: IntParam = new IntParam(this, "dmlcWorkerConnectRetry",
    "number of retry worker do before fail", ParamValidators.gtEq(1))

  final def exitOnError: BooleanParam = new BooleanParam(this, "dmlcWorkerStopProcessOnError",
  "exit process when rabit see assert error")

  setDefault(ringReduce -> false, reduceBuffer -> "256MB", debug -> false,
    connectRetry -> 5, timeout -> -1, exitOnError -> false)
}
