package ml.dmlc.xgboost4j.java;

import java.io.Serializable;
import java.util.Iterator;

import ml.dmlc.xgboost4j.LabeledPoint;

/**
 * A mini-batch of data that can be converted to DMatrix.
 * The data is in sparse matrix CSR format.
 *
 * This class is used to support advanced creation of DMatrix from Iterator of DataBatch,
 */
class DataBatch {
  /** The offset of each rows in the sparse matrix */
  long[] rowOffset = null;
  /** weight of each data point, can be null */
  float[] weight = null;
  /** label of each data point, can be null */
  float[] label = null;
  /** index of each feature(column) in the sparse matrix */
  int[] featureIndex = null;
  /** value of each non-missing entry in the sparse matrix */
  float[] featureValue = null;

  public DataBatch() {}

  public DataBatch(long[] rowOffset, float[] weight, float[] label, int[] featureIndex,
                   float[] featureValue) {
    this.rowOffset = rowOffset;
    this.weight = weight;
    this.label = label;
    this.featureIndex = featureIndex;
    this.featureValue = featureValue;
  }


  /**
   * Get number of rows in the data batch.
   * @return Number of rows in the data batch.
   */
  public int numRows() {
    return rowOffset.length - 1;
  }

  /**
   * Shallow copy a DataBatch
   * @return a copy of the batch
   */
  public DataBatch shallowCopy() {
    DataBatch b = new DataBatch();
    b.rowOffset = this.rowOffset;
    b.weight = this.weight;
    b.label = this.label;
    b.featureIndex = this.featureIndex;
    b.featureValue = this.featureValue;
    return b;
  }

  static class BatchIterator implements Iterator<DataBatch> {
    private Iterator<LabeledPoint> base;
    private int batchSize;

    BatchIterator(java.util.Iterator<LabeledPoint> base, int batchSize) {
      this.base = base;
      this.batchSize = batchSize;
    }
    @Override
    public boolean hasNext() {
      return base.hasNext();
    }
    @Override
    public DataBatch next() {
      int num_rows = 0, num_elem = 0;
      java.util.List<LabeledPoint> batch = new java.util.ArrayList<LabeledPoint>();
      for (int i = 0; i < this.batchSize; ++i) {
        if (!base.hasNext()) break;
        LabeledPoint inst = base.next();
        batch.add(inst);
        num_elem += inst.values.length;
        ++num_rows;
      }
      DataBatch ret = new DataBatch();
      // label
      ret.rowOffset = new long[num_rows + 1];
      ret.label = new float[num_rows];
      ret.featureIndex = new int[num_elem];
      ret.featureValue = new float[num_elem];
      // current offset
      int offset = 0;
      for (int i = 0; i < batch.size(); ++i) {
        LabeledPoint inst = batch.get(i);
        ret.rowOffset[i] = offset;
        ret.label[i] = inst.label;
        if (inst.indices != null) {
          System.arraycopy(inst.indices, 0, ret.featureIndex, offset, inst.indices.length);
        } else{
          for (int j = 0; j < inst.values.length; ++j) {
            ret.featureIndex[offset + j] = j;
          }
        }
        System.arraycopy(inst.values, 0, ret.featureValue, offset, inst.values.length);
        offset += inst.values.length;
      }
      ret.rowOffset[batch.size()] = offset;
      return ret;
    }
    @Override
    public void remove() {
      throw new Error("not implemented");
    }
  }
}
