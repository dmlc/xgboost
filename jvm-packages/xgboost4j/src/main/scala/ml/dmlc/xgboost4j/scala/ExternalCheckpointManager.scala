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

package ml.dmlc.xgboost4j.scala

import ml.dmlc.xgboost4j.java.{ExternalCheckpointManager => JavaECM}
import org.apache.hadoop.fs.FileSystem

class ExternalCheckpointManager(checkpointPath: String, fs: FileSystem)
  extends JavaECM(checkpointPath, fs) {

  def updateCheckpoint(booster: Booster): Unit = {
    super.updateCheckpoint(booster.booster)
  }

  def loadCheckpointAsScalaBooster(): Booster = {
    val loadedBooster = super.loadCheckpointAsBooster()
    if (loadedBooster == null) {
      null
    } else {
      new Booster(loadedBooster)
    }
  }
}
