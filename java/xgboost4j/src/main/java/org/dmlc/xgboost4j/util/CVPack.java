/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.dmlc.xgboost4j.util;

import java.util.Map;
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
    
    public CVPack(DMatrix dtrain, DMatrix dtest, Map<String, String> params) {
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
