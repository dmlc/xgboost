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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author hzx
 */
public class PredictExample {
    public static void main(String[] args) throws IOException {
        //params
        Map<String, String> param = new HashMap<String, String>() {
            {
                put("silent", "1");
                put("nthread", "6");
                put("num_class", "9");
            }
        };
        
        //load model from file
        String modelPath = "./tmp/xgb_model";
        Booster booster = new Booster(param, modelPath);
        //get test DMatrix
        String testPath = "./tmp/final_test.txt";
        DMatrix tmat = new DMatrix(testPath);
        
        //predict
        float[][] predicts = booster.predict(tmat);
        
        //check predicts
        for(float[] pArray : predicts) {
            System.out.println(Arrays.toString(pArray));
        }
        
        //release
        tmat.delete();
        booster.delete();
    }
}
