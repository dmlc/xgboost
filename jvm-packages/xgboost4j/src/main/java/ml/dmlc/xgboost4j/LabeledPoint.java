package ml.dmlc.xgboost4j;

/**
 * Labeled data point for training examples.
 * Represent a sparse training instance.
 */
public class LabeledPoint {
  /** Label of the point */
  float label;
  /** Weight of this data point */
  float weight = 1.0f;
  /** Feature indices, used for sparse input */
  int[] indices = null;
  /** Feature values */
  float[] values;

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
