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
package org.dmlc.xgboost4j.demo.util;

import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IEvaluation;

/**
 * a util evaluation class for examples
 * @author hzx
 */
public class CustomEval implements IEvaluation {

    String evalMetric = "custom_error";
        
    @Override
    public String getMetric() {
        return evalMetric;
    }

    @Override
    public float eval(float[][] predicts, DMatrix dmat) {
        float error = 0f;
        float[] labels = dmat.getLabel();
        int nrow = predicts.length;
        for(int i=0; i<nrow; i++) {
            if(labels[i]==0f && predicts[i][0]>0.5) {
                error++;
            }
            else if(labels[i]==1f && predicts[i][0]<=0.5) {
                error++;
            }
        }
            
        return error/labels.length;
    }
}
