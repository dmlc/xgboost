package ml.dmlc.xgboost4j.java;

public class DistributedEvalError implements IEvaluation, IEvaluationForDistributed {

  @Override
  public float evalRow(float label, float pred) {
    System.out.println("aaa");
    return 1.0f;
  }

  @Override
  public float getFinal(float errorSum, float weightSum) {
    return errorSum/weightSum;
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
