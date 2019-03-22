package ml.dmlc.xgboost4j.java;

public class DistributedEvalError implements IEvaluationForDistributed, IEvaluation {

  @Override
  public float evalRow(float label, float pred) {
    return 1.0f;
  }

  @Override
  public float getFinal(float errorSum, float weightSum) {
    return 1.0f;
  }

  @Override
  public float constant() {
    return 0;
  }

  @Override
  public float constant1(float e) {
    return e;
  }

  @Override
  public boolean hasNext() {
    return false;
  }

  @Override
  public String getMetric() {
    return "distributed_error";
  }

  @Override
  public float eval(float[][] predicts, DMatrix dmat) {
    return 0;
  }
}
