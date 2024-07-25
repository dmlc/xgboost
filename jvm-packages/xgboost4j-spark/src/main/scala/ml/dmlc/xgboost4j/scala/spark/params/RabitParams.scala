/*
 Copyright (c) 2014-2022 by Contributors

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

private[spark] trait RabitParams extends Params with NonXGBoostParams {

  final val rabitTrackerTimeout = new IntParam(this, "rabitTrackerTimeout", "The number of " +
    "seconds before timeout waiting for workers to connect. and for the tracker to shutdown.",
    ParamValidators.gtEq(0))

  final def getRabitTrackerTimeout: Int = $(rabitTrackerTimeout)

  final val rabitTrackerHostIp = new Param[String](this, "rabitTrackerHostIp", "The Rabit " +
    "Tracker host IP address. This is only needed if the host IP cannot be automatically " +
    "guessed.")

  final def getRabitTrackerHostIp: String = $(rabitTrackerHostIp)

  final val rabitTrackerPort = new IntParam(this, "rabitTrackerPort", "The port number for the " +
    "tracker to listen to. Use a system allocated one by default.",
    ParamValidators.gtEq(0))

  final def getRabitTrackerPort: Int = $(rabitTrackerPort)

  setDefault(rabitTrackerTimeout -> 0, rabitTrackerHostIp -> "", rabitTrackerPort -> 0)

  addNonXGBoostParam(rabitTrackerPort, rabitTrackerHostIp, rabitTrackerPort)
}
