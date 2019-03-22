package ml.dmlc.xgboost4j.java;

public interface IEvaluationForDistributed {

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  float evalRow(float label, float pred);

  /**
   * perform transformation with the sum of error and weights to get the final evaluation metrics
   */
  float getFinal(float errorSum, float weightSum);

  float constant();

  float constant1(float e);

  boolean hasNext();
}
