/*
 Copyright (c) 2021-2024 by Contributors

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

package ml.dmlc.xgboost4j.java;

/**
 * This Column abstraction provides an array interface JSON string, which is
 * used to reconstruct columnar data within the XGBoost library.
 */
public abstract class Column implements AutoCloseable {

  /**
   * Return array interface json string for this Column
   */
  public abstract String toJson();

  @Override
  public void close() throws Exception {
  }
}
