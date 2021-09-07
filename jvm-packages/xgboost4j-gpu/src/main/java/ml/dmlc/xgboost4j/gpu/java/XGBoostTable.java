package ml.dmlc.xgboost4j.gpu.java;

// Interface to build DMatrix on device.
public abstract class XGBoostTable implements AutoCloseable {

  /** Get the cuda array interface json string for the whole table */
  public abstract String getArrayInterfaceJson();

  /**
   * Get the cuda array interface of the feature columns.
   * The turned value must not be null or empty
   */
  public abstract String getFeatureArrayInterface();

  @Override
  public void close() throws Exception {

  }
}
