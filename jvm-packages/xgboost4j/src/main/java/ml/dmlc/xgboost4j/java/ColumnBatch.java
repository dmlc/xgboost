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
 * The abstracted XGBoost ColumnBatch to get cuda array interface which is used to build
 * the DMatrix on device.
 *
 */
public abstract class ColumnBatch implements AutoCloseable {

  /**
   * Get the cuda array interface json string for the whole ColumnBatch including
   * the must-have feature, label columns and the optional weight, base margin columns.
   *
   * This API will be called by {@link DMatrix#DMatrix(Iterator, float, int, int)}
   *
   */
  public final String getArrayInterfaceJson() {
    StringBuilder builder = new StringBuilder();
    builder.append("{");
    String featureStr = this.getFeatureArrayInterface();
    if (featureStr == null || featureStr.isEmpty()) {
      throw new RuntimeException("Feature json must not be empty");
    } else {
      builder.append("\"features_str\":" + featureStr);
    }

    String labelStr = this.getLabelsArrayInterface();
    if (labelStr != null && ! labelStr.isEmpty()) {
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
   *
   * This API will be called by {@link DMatrix#DMatrix(ColumnBatch, float, int)}
   */
  public abstract String getFeatureArrayInterface();

  public abstract String getLabelsArrayInterface();

  public abstract String getWeightsArrayInterface();

  public abstract String getBaseMarginsArrayInterface();

  @Override
  public void close() throws Exception {}
}
