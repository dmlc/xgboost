package ml.dmlc.xgboost4j.java;

public interface IEvalElementWiseDistributed extends IEvaluation {

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  float evalRow(float label, float pred);

  /**
   * perform transformation with the sum of error and weights to get the final evaluation metrics
   */
  float getFinal(float errorSum, float weightSum);

  @Override
  default float eval(float[][] predicts, DMatrix dmat) {
    throw new RuntimeException("IEvalElementWiseDistributed does not support eval method");
  }
}
