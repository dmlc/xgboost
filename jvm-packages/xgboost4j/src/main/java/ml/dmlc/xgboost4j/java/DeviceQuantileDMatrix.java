package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

public class DeviceQuantileDMatrix extends DMatrix {
  /**
   * Create DeviceQuantileDMatrix from iterator based on the cuda array interface
   * @param iter the XGBoost ColumnBatch batch to provide the corresponding cuda array interface
   * @param missing the missing value
   * @param maxBin the max bin
   * @param nthread the parallelism
   * @throws XGBoostError
   */
  public DeviceQuantileDMatrix(
      Iterator<ColumnBatch> iter, float missing, int maxBin, int nthread
  ) throws XGBoostError {
    super(0);
    long[] out = new long[1];
    Iterator<DataFrameBatch> batchIter = new DataFrameBatch.BatchIterator(iter);
    XGBoostJNI.checkCall(XGBoostJNI.XGDeviceQuantileDMatrixCreateFromCallback(
        batchIter, missing, maxBin, nthread, out));
    handle = out[0];
  }

  @Override
  public void setLabel(Column column) throws XGBoostError {
    throw new XGBoostError("Not applicable.");
  }
  @Override
  public void setWeight(Column column) throws XGBoostError {
    throw new XGBoostError("Not applicable.");
  }
}
