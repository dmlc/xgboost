/*
 Copyright (c) 2014-2025 by Contributors

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
package ml.dmlc.xgboost4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Labeled training data point.
 * TODO(hcho3): Migrate Record class when we upgrade to Java 14+, to reduce boilerplate.
 */
public final class LabeledPoint implements Serializable {
  private final float label;
  private final int size;
  private final int[] indices;
  private final float[] values;
  private final float weight;
  private final int group;
  private final float baseMargin;

  /**
   * @param label Label of this point.
   * @param size Feature dimensionality
   * @param indices Feature indices of this point or `null` if the data is dense.
   * @param values Feature values of this point.
   * @param weight Weight of this point.
   * @param group Group of this point (used for ranking) or -1.
   * @param baseMargin Initial prediction on this point or `Float.NaN`
   */
  public LabeledPoint(
      float label, int size, int[] indices, float[] values, float weight,
      int group, float baseMargin
  ) {
    assert (indices == null || indices.length == values.length):
      "indices and values must have the same number of elements";
    assert (indices == null || size >= indices.length):
      "feature dimensionality must be greater equal than size of indices";
    this.label = label;
    this.size = size;
    this.indices = indices;
    this.values = values;
    this.weight = weight;
    this.group = group;
    this.baseMargin = baseMargin;
  }

  /**
   * @param label Label of this point.
   * @param size Feature dimensionality
   * @param indices Feature indices of this point or `null` if the data is dense.
   * @param values Feature values of this point.
   */
  public LabeledPoint(
      float label, int size, int[] indices, float[] values
  ) {
    this(label, size, indices, values, 1.0f, -1, Float.NaN);
  }

  /**
   * @param label Label of this point.
   * @param size Feature dimensionality
   * @param indices Feature indices of this point or `null` if the data is dense.
   * @param values Feature values of this point.
   * @param weight Weight of this point.
   */
  public LabeledPoint(
      float label, int size, int[] indices, float[] values, float weight
  ) {
    this(label, size, indices, values, weight, -1, Float.NaN);
  }

  /**
   * @param label Label of this point.
   * @param size Feature dimensionality
   * @param indices Feature indices of this point or `null` if the data is dense.
   * @param values Feature values of this point.
   * @param weight Weight of this point.
   * @param group Group of this point (used for ranking) or -1.
   */
  public LabeledPoint(
      float label, int size, int[] indices, float[] values, float weight,
      int group
  ) {
    this(label, size, indices, values, weight, group, Float.NaN);
  }


  @Override
  public int hashCode() {
    return Objects.hash(this.label, this.size, Arrays.hashCode(this.indices),
      Arrays.hashCode(this.values), this.weight, this.group, this.baseMargin);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    } else if (!(obj instanceof LabeledPoint)) {
      return false;
    } else {
      LabeledPoint other = (LabeledPoint) obj;
      return Objects.equals(label, other.label)
        && Objects.equals(size, other.size)
        && Arrays.equals(indices, other.indices)
        && Arrays.equals(values, other.values)
        && Objects.equals(weight, other.weight)
        && Objects.equals(group, other.group)
        && Objects.equals(baseMargin, other.baseMargin);
    }
  }

  public float label() { return this.label; }
  public int size() { return this.size; }
  public int[] indices() { return this.indices; }
  public float[] values() { return this.values; }
  public float weight() { return this.weight; }
  public int group() { return this.group; }
  public float baseMargin() { return this.baseMargin; }
}
