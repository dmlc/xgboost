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
import java.util.Arrays;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.util.Params;

/**
 *
 * @author hzx
 */
public class PredictExample {
    public static void main(String[] args) throws IOException {
        //params
        Params param = new Params() {
            {
                put("silent", "1");
                put("nthread", "6");
                put("num_class", "9");
            }
        };
        
        //load model from file
        String modelPath = "./data/xgb_model.bin";
        Booster booster = new Booster(param, modelPath);
        //get test DMatrix
        String testPath = "./data/test.txt";
        DMatrix tmat = new DMatrix(testPath);
        
        //predict
        float[][] predicts = booster.predict(tmat);
        
        //check predicts
        for(float[] pArray : predicts) {
            System.out.println(Arrays.toString(pArray));
        }
        
        //predict leaf
        float[][] leaf_predicts = booster.predict(tmat, 0, true);
        
        for(float[] leafs : leaf_predicts) {
            System.out.println(Arrays.toString(leafs));
        }
    }
}
