/*
 Copyright (c) 2021 by Contributors

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
 * The abstracted XGBoost DataFrame to build the DMatrix on device.
 *
 */
public abstract class DataFrame implements AutoCloseable {

  /** Get the cuda array interface json string for the whole DataFrame */
  public abstract String getArrayInterfaceJson();

  /**
   * Get the cuda array interface of the feature columns.
   * The turned value must not be null or empty
   */
  public abstract String getFeatureArrayInterface();

  /**
   * Get the cuda array interface of the label column.
   * The turned value can be null or empty
   */
  public abstract String getLabelArrayInterface();

  /**
   * Get the cuda array interface of the weight column.
   * The turned value can be null or empty
   */
  public abstract String getWeightArrayInterface();

  /**
   * Get the cuda array interface of the base margin column.
   * The turned value can be null or empty
   */
  public abstract String getBaseMarginArrayInterface();

  @Override
  public void close() throws Exception {}

}
