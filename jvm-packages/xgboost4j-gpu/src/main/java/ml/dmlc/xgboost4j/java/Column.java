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
 * The abstracted XGBoost Column to get the cuda array interface which is used to
 * set the information for DMatrix.
 *
 */
public abstract class Column implements AutoCloseable {

  /**
   * Get the cuda array interface json string for the Column which can be representing
   * weight, label, base margin column.
   *
   * This API will be called by
   *  {@link DMatrix#setLabel(Column)}
   *  {@link DMatrix#setWeight(Column)}
   *  {@link DMatrix#setBaseMargin(Column)}
   */
  public abstract String getArrayInterfaceJson();

  @Override
  public void close() throws Exception {}

}
