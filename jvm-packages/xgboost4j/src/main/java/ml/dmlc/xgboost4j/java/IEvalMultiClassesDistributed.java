package ml.dmlc.xgboost4j.java;

public interface IEvalMultiClassesDistributed extends IEvaluation {

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  float evalRow(int label, float pred, int numClasses);

  /**
   * perform transformation with the sum of error and weights to get the final evaluation metrics
   */
  default float getFinal(float errorSum, float weightSum) {
    return errorSum / weightSum;
  }

  @Override
  default float eval(float[][] predicts, DMatrix dmat) {
    throw new RuntimeException("IEvalMultiClassesDistributed does not support eval method");
  }

}
