/*
 Copyright (c) 2014 by Contributors 

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
package org.dmlc.xgboost4j.demo;

import java.io.IOException;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.util.XGBoostError;

/**
 * an example of cross validation
 * @author hzx
 */
public class CrossValidation {
    public static void main(String[] args) throws IOException, XGBoostError {
         //load train mat
        DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        
        //set params
        Params param = new Params() {
            {
                put("eta", 1.0);
                put("max_depth", 3);
                put("silent", 1);
                put("nthread", 6);
                put("objective", "binary:logistic");
                put("gamma", 1.0);
                put("eval_metric", "error");
            }
        };
        
        //do 5-fold cross validation
        int round = 2;
        int nfold = 5;
        //set additional eval_metrics
        String[] metrics = null;
        
        String[] evalHist = Trainer.crossValiation(param, trainMat, round, nfold, metrics, null, null);
    }
}
