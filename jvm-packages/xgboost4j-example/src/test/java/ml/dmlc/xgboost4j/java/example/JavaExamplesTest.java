/*
 Copyright (c) 2024 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.java.example;

import java.io.IOException;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.junit.Test;


public class JavaExamplesTest {

  @Test
  public void testExamples() throws XGBoostError, IOException {
    String[] args = {""};
    System.out.println("BasicWalkThrough");
    BasicWalkThrough.main(args);
    System.out.println("BoostFromPrediction");
    BoostFromPrediction.main(args);
    System.out.println("CrossValidation");
    CrossValidation.main(args);
    System.out.println("CustomObjective");
    CustomObjective.main(args);
    System.out.println("EarlyStopping");
    EarlyStopping.main(args);
    System.out.println("ExternalMemory");
    ExternalMemory.main(args);
    System.out.println("GeneralizedLinearModel");
    GeneralizedLinearModel.main(args);
    System.out.println("PredictFirstNtree");
    PredictFirstNtree.main(args);
    System.out.println("PredictLeafIndices");
    PredictLeafIndices.main(args);
  }
}
