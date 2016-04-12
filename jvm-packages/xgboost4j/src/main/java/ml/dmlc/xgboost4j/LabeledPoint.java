package ml.dmlc.xgboost4j;

import java.io.Serializable;

/**
 * Labeled data point for training examples.
 * Represent a sparse training instance.
 */
public class LabeledPoint implements Serializable {
  /** Label of the point */
  public float label;
  /** Weight of this data point */
  public float weight = 1.0f;
  /** Feature indices, used for sparse input */
  public int[] indices = null;
  /** Feature values */
  public float[] values;

  private LabeledPoint() {}

  /**
   * Create Labeled data point from sparse vector.
   * @param label The label of the data point.
   * @param indices The indices
   * @param values The values.
   */
  public static LabeledPoint fromSparseVector(float label, int[] indices, float[] values) {
    LabeledPoint ret = new LabeledPoint();
    ret.label = label;
    ret.indices = indices;
    ret.values = values;
    assert indices.length == values.length;
    return ret;
  }

  /**
   * Create Labeled data point from dense vector.
   * @param label The label of the data point.
   * @param values The values.
   */
  public static LabeledPoint fromDenseVector(float label, float[] values) {
    LabeledPoint ret = new LabeledPoint();
    ret.label = label;
    ret.indices = null;
    ret.values = values;
    return ret;
  }
}
