package ml.dmlc.xgboost4j.java;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import ml.dmlc.xgboost4j.LabeledPoint;

/**
 * A mini-batch of data that can be converted to DMatrix.
 * The data is in sparse matrix CSR format.
 *
 * This class is used to support advanced creation of DMatrix from Iterator of DataBatch,
 */
class DataBatch {
  private static final Log logger = LogFactory.getLog(DataBatch.class);
  /** The offset of each rows in the sparse matrix */
  final long[] rowOffset;
  /** weight of each data point, can be null */
  final float[] weight;
  /** label of each data point, can be null */
  final float[] label;
  /** index of each feature(column) in the sparse matrix */
  final int[] featureIndex;
  /** value of each non-missing entry in the sparse matrix */
  final float[] featureValue ;

  DataBatch(long[] rowOffset, float[] weight, float[] label, int[] featureIndex,
            float[] featureValue) {
    this.rowOffset = rowOffset;
    this.weight = weight;
    this.label = label;
    this.featureIndex = featureIndex;
    this.featureValue = featureValue;
  }

  static class BatchIterator implements Iterator<DataBatch> {
    private final Iterator<LabeledPoint> base;
    private final int batchSize;

    BatchIterator(Iterator<LabeledPoint> base, int batchSize) {
      this.base = base;
      this.batchSize = batchSize;
    }

    @Override
    public boolean hasNext() {
      return base.hasNext();
    }

    @Override
    public DataBatch next() {
      try {
        int numRows = 0;
        int numElem = 0;
        List<LabeledPoint> batch = new ArrayList<>(batchSize);
        while (base.hasNext() && batch.size() < batchSize) {
          LabeledPoint labeledPoint = base.next();
          batch.add(labeledPoint);
          numElem += labeledPoint.values().length;
          numRows++;
        }

        long[] rowOffset = new long[numRows + 1];
        float[] label = new float[numRows];
        int[] featureIndex = new int[numElem];
        float[] featureValue = new float[numElem];
        float[] weight = new float[numRows];

        int offset = 0;
        for (int i = 0; i < batch.size(); i++) {
          LabeledPoint labeledPoint = batch.get(i);
          rowOffset[i] = offset;
          label[i] = labeledPoint.label();
          weight[i] = labeledPoint.weight();
          if (labeledPoint.indices() != null) {
            System.arraycopy(labeledPoint.indices(), 0, featureIndex, offset,
                    labeledPoint.indices().length);
          } else {
            for (int j = 0; j < labeledPoint.values().length; j++) {
              featureIndex[offset + j] = j;
            }
          }

          System.arraycopy(labeledPoint.values(), 0, featureValue, offset,
                  labeledPoint.values().length);
          offset += labeledPoint.values().length;
        }

        rowOffset[batch.size()] = offset;
        return new DataBatch(rowOffset, weight, label, featureIndex, featureValue);
      } catch (RuntimeException runtimeError) {
        logger.error(runtimeError);
        return null;
      }
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException("DataBatch.BatchIterator.remove");
    }
  }
}
