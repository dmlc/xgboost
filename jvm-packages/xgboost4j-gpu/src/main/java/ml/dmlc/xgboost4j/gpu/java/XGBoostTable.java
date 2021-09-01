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

  /**
   * Get the cuda array interface of the label column.
   * The turned value must not be null or empty
   */
  public abstract String getLabelArrayInterface();

  /**
   * Get the cuda array interface of the weight column.
   * The turned value can be null or empty
   */
  public abstract String getWeightArrayInterface();

  /**
   * Get the cuda array interface of the base margin column.
   * The turned value can be null or empty
   */
  public abstract String getBaseMarginArrayInterface();

  @Override
  public void close() throws Exception {

  }
}
