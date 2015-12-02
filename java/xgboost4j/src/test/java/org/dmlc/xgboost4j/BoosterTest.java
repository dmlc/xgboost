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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import junit.framework.TestCase;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;
import org.junit.Test;

/**
 * test cases for Booster
 * @author hzx
 */
public class BoosterTest {
    public static class EvalError implements IEvaluation {
        private static final Log logger = LogFactory.getLog(EvalError.class);
        
        String evalMetric = "custom_error";
        
        public EvalError() {
        }
        
        @Override
        public String getMetric() {
            return evalMetric;
        }

        @Override
        public float eval(float[][] predicts, DMatrix dmat) {
            float error = 0f;
            float[] labels;
            try {
                labels = dmat.getLabel();
            } catch (XGBoostError ex) {
                logger.error(ex);
                return -1f;
            }
            int nrow = predicts.length;
            for(int i=0; i<nrow; i++) {
                if(labels[i]==0f && predicts[i][0]>0) {
                    error++;
                }
                else if(labels[i]==1f && predicts[i][0]<=0) {
                    error++;
                }
            }
            
            return error/labels.length;
        }
    }
    
    @Test
    public void testBoosterBasic() throws XGBoostError {
        DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
        
        //set params
        Map<String, Object> paramMap = new HashMap<String, Object>() {
            {
                put("eta", 1.0);
                put("max_depth", 2);
                put("silent", 1);
                put("objective", "binary:logistic");
            }
        };
        Iterable<Entry<String, Object>> param = paramMap.entrySet();
        
        //set watchList
        List<Entry<String, DMatrix>> watchs =  new ArrayList<>();
        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));

         //set round
        int round = 2;
        
        //train a boost model
        Booster booster = Trainer.train(param, trainMat, round, watchs, null, null);
        
         //predict raw output
        float[][] predicts = booster.predict(testMat, true);
        
        //eval
        IEvaluation eval = new EvalError();
        //error must be less than 0.1
        TestCase.assertTrue(eval.eval(predicts, testMat)<0.1f);
    }
}
