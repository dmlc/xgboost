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

import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;

/**
 * cross validation package for xgb
 * @author hzx
 */
public class CVPack {
    DMatrix dtrain;
    DMatrix dtest;
    long[] dataArray;
    String[] names;
    Booster booster;
    
    public CVPack(DMatrix dtrain, DMatrix dtest, Params params) {
        DMatrix[] dmats = new DMatrix[] {dtrain, dtest};
        booster = new Booster(params, dmats);
        dataArray = TransferUtil.dMatrixs2handles(dmats);
        names = new String[] {"train", "test"};
        this.dtrain = dtrain;
        this.dtest = dtest;
    }
    
    public void update(int iter) {
        booster.update(dtrain, iter);
    }
    
    public String eval(int iter) {
        return booster.evalSet(dataArray, names, iter);
    }
}
