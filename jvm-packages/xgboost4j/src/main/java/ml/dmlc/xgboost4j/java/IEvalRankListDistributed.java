package ml.dmlc.xgboost4j.java;

public interface IEvalRankListDistributed extends IEvaluation {

  float evalMetric(float[] preds, int[] labels);

  @Override
  default float eval(float[][] predicts, DMatrix dmat) {
    throw new RuntimeException("IEvalRankingDistributed does not support eval method");
  }
}
