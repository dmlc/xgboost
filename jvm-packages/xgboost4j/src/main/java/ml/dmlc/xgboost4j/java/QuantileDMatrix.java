package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

/**
 * QuantileDMatrix will only be used to train
 */
public class QuantileDMatrix extends DMatrix {
  /**
   * Create QuantileDMatrix from iterator based on the cuda array interface
   *
   * @param iter    the XGBoost ColumnBatch batch to provide the corresponding cuda array interface
   * @param missing the missing value
   * @param maxBin  the max bin
   * @param nthread the parallelism
   * @throws XGBoostError
   */
  public QuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      int nthread) throws XGBoostError {
    super(0);
    long[] out = new long[1];
    String conf = getConfig(missing, maxBin, nthread);
    XGBoostJNI.checkCall(XGBoostJNI.XGQuantileDMatrixCreateFromCallback(
        iter, (java.util.Iterator<ColumnBatch>)null, conf, out));
    handle = out[0];
  }

  @Override
  public void setLabel(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setLabel.");
  }

  @Override
  public void setWeight(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setWeight.");
  }

  @Override
  public void setBaseMargin(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setLabel(float[] labels) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setLabel.");
  }

  @Override
  public void setWeight(float[] weights) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setWeight.");
  }

  @Override
  public void setBaseMargin(float[] baseMargin) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setBaseMargin(float[][] baseMargin) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setGroup(int[] group) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setGroup.");
  }

  private String getConfig(float missing, int maxBin, int nthread) {
    return String.format("{\"missing\":%f,\"max_bin\":%d,\"nthread\":%d}",
        missing, maxBin, nthread);
  }
}
