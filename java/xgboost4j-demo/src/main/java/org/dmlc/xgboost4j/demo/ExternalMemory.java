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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

/**
 * simple example for using external memory version
 * @author hzx
 */
public class ExternalMemory {
    public static void main(String[] args) throws XGBoostError {
        //this is the only difference, add a # followed by a cache prefix name
        //several cache file with the prefix will be generated
        //currently only support convert from libsvm file
        DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train#dtrain.cache");
        DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test#dtest.cache");
        
        //specify parameters
        Params param = new Params() {
            {
                put("eta", 1.0);
                put("max_depth", 2);
                put("silent", 1);
                put("objective", "binary:logistic");
            }
        };
        
        //performance notice: set nthread to be the number of your real cpu
        //some cpu offer two threads per core, for example, a 4 core cpu with 8 threads, in such case set nthread=4
        //param.put("nthread", num_real_cpu);
        
        //specify watchList
        List<Map.Entry<String, DMatrix>> watchs =  new ArrayList<>();
        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));
        
        //set round
        int round = 2;
        
        //train a boost model
        Booster booster = Trainer.train(param, trainMat, round, watchs, null, null);
    }
}
