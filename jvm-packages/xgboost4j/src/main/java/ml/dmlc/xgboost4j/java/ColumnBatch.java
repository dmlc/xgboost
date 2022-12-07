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

import java.util.Iterator;

/**
 * The abstracted XGBoost ColumnBatch to get array interface from columnar data format.
 * For example, the cuDF dataframe which employs apache arrow specification.
 */
public abstract class ColumnBatch implements AutoCloseable {
  /**
   * Get the cuda array interface json string for the whole ColumnBatch including
   * the must-have feature, label columns and the optional weight, base margin columns.
   *
   * This function is be called by native code during iteration and can be made as private
   * method.  We keep it as public simply to silent the linter.
   */
  public final String getArrayInterfaceJson() {

    StringBuilder builder = new StringBuilder();
    builder.append("{");
    String featureStr = this.getFeatureArrayInterface();
    if (featureStr == null || featureStr.isEmpty()) {
      throw new RuntimeException("Feature array interface must not be empty");
    } else {
      builder.append("\"features_str\":" + featureStr);
    }

    String labelStr = this.getLabelsArrayInterface();
    if (labelStr == null || labelStr.isEmpty()) {
      throw new RuntimeException("Label array interface must not be empty");
    } else {
      builder.append(",\"label_str\":" + labelStr);
    }

    String weightStr = getWeightsArrayInterface();
    if (weightStr != null && ! weightStr.isEmpty()) {
      builder.append(",\"weight_str\":" + weightStr);
    }

    String baseMarginStr = getBaseMarginsArrayInterface();
    if (baseMarginStr != null && ! baseMarginStr.isEmpty()) {
      builder.append(",\"basemargin_str\":" + baseMarginStr);
    }

    builder.append("}");
    return builder.toString();
  }

  /**
   * Get the cuda array interface of the feature columns.
   * The returned value must not be null or empty
   */
  public abstract String getFeatureArrayInterface();

  /**
   * Get the cuda array interface of the label columns.
   * The returned value must not be null or empty if we're creating
   *  {@link QuantileDMatrix#QuantileDMatrix(Iterator, float, int, int)}
   */
  public abstract String getLabelsArrayInterface();

  /**
   * Get the cuda array interface of the weight columns.
   * The returned value can be null or empty
   */
  public abstract String getWeightsArrayInterface();

  /**
   * Get the cuda array interface of the base margin columns.
   * The returned value can be null or empty
   */
  public abstract String getBaseMarginsArrayInterface();

  @Override
  public void close() throws Exception {}

}
