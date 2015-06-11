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
package org.dmlc.xgboost4j.util;

import java.util.Map;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IObjective;

/**
 * cross validation package for xgb
 * @author hzx
 */
public class CVPack {
    DMatrix dtrain;
    DMatrix dtest;
    DMatrix[] dmats;
    String[] names;
    Booster booster;
    
    /**
     * create an cross validation package
     * @param dtrain train data
     * @param dtest test data
     * @param params parameters
     */
    public CVPack(DMatrix dtrain, DMatrix dtest, Iterable<Map.Entry<String, Object>> params) {
        dmats = new DMatrix[] {dtrain, dtest};
        booster = new Booster(params, dmats);
        names = new String[] {"train", "test"};
        this.dtrain = dtrain;
        this.dtest = dtest;
    }
    
    /**
     * update one iteration
     * @param iter iteration num
     */
    public void update(int iter) {
        booster.update(dtrain, iter);
    }
    
    /**
     * update one iteration
     * @param iter iteration num
     * @param obj customized objective
     */
    public void update(int iter, IObjective obj) {
        booster.update(dtrain, iter, obj);
    }
    
    /**
     * evaluation 
     * @param iter iteration num
     * @return 
     */
    public String eval(int iter) {
        return booster.evalSet(dmats, names, iter);
    }
    
    /**
     * evaluation 
     * @param iter iteration num
     * @param eval customized eval
     * @return 
     */
    public String eval(int iter, IEvaluation eval) {
        return booster.evalSet(dmats, names, iter, eval);
    }
}
