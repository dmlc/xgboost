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
package org.dmlc.xgboost4j;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author hzx
 */
public class TrainExample {
    public static void main(String[] args) throws IOException {
        //load train mat
        DMatrix trainMat = new DMatrix("./tmp/final_train.txt");
        //load valid mat
        DMatrix validMat = new DMatrix("./tmp/final_valid.txt");
        
        //set params
        Map<String, String> param = new HashMap<String, String>() {
            {
                put("eta", "0.1");
                put("max_depth", "8");
                put("silent", "1");
                put("nthread", "6");
                put("num_class", "9");
                put("objective", "multi:softprob");
                put("eval_metric", "mlogloss");
            }
        };
        
        //initialize booster
        DMatrix[] dmats = new DMatrix[] {trainMat, validMat};
        Booster booster = new Booster(param, dmats);
        
        String[] evalNames = new String[] {"train", "valid"};
        //train booster
        int round = 300;
        for(int iter=0; iter<round; iter++) {
            booster.update(trainMat, iter);
            
            //evaluation
            String evalInfo = booster.evalSet(dmats, evalNames, iter);
            System.out.println(evalInfo);
        }
        
        //save model
        booster.saveModel("./tmp/xgb_model");
        
        //release all
        trainMat.delete();
        validMat.delete();
    }
}
