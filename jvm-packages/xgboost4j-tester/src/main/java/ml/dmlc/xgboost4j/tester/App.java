package ml.dmlc.xgboost4j.tester;

import ml.dmlc.xgboost4j.java.example.*;

import java.io.IOException;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class App {
  public static void main(String[] args) throws IOException, XGBoostError {
    String[] args2 = new String[0];
    System.out.println("BoostFromPrediction");
    BoostFromPrediction.main(args2);
    System.out.println("CrossValidation");
    CrossValidation.main(args2);
    System.out.println("CustomObjective");
    CustomObjective.main(args2);
    System.out.println("ExternalMemory");
    ExternalMemory.main(args2);
    System.out.println("GeneralizedLinearModel");
    GeneralizedLinearModel.main(args2);
    System.out.println("PredictFirstNtree");
    PredictFirstNtree.main(args2);
    System.out.println("PredictLeafIndices");
    PredictLeafIndices.main(args2);
  }
}
