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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;


/**
 * trainer for xgboost
 * @author hzx
 */
public class Trainer {
    private static final Log logger = LogFactory.getLog(Trainer.class);
    
    /**
     * Train a booster with given parameters.
     * @param params Booster params.
     * @param dtrain Data to be trained.
     * @param round Number of boosting iterations.
     * @param evalMats Data to be evaluated (may include dtrain)
     * @param evalNames name of data (used for evaluation info)
     * @return 
     */
    public static Booster train(Map<String, String> params, DMatrix dtrain, int round,
            DMatrix[] evalMats, String[] evalNames) {
        //collect all data matrixs
        DMatrix[] allMats;
        if(evalMats!=null && evalMats.length>0) {
            allMats = new DMatrix[evalMats.length+1];
            allMats[0] = dtrain;
            System.arraycopy(evalMats, 0, allMats, 1, evalMats.length);
        }
        else {
            allMats = new DMatrix[1];
            allMats[0] = dtrain;
        }
        
        //initialize booster
        Booster booster = new Booster(params, allMats);
        
        //used for evaluation
        long[] dataArray = null;
        String[] names = null;
        
        //begin to train
        for(int iter=0; iter<round; iter++) {
            booster.update(dtrain, iter);
            
            //evaluation
            if(evalMats!=null && evalMats.length>0) {
                if(dataArray==null || names==null) {
                    //prepare data for evaluation
                    dataArray = TransferUtil.dMatrixs2handles(evalMats);
                    names = evalNames;
                } 
                String evalInfo = booster.evalSet(dataArray, names, iter);
                logger.info(evalInfo);
            }
        }
        return booster;
    }
    
    /**
     * Cross-validation with given paramaters.
     * @param params Booster params.
     * @param data  Data to be trained.
     * @param round Number of boosting iterations.
     * @param nfold Number of folds in CV.
     * @param metrics Evaluation metrics to be watched in CV.
     * @return evaluation history
     */
    public static String[] crossValiation(Map<String, String> params, DMatrix data, int round, int nfold, String[] metrics) {
        CVPack[] cvPacks = makeNFold(data, nfold, params, metrics);
        String[] evalHist = new String[round];
        String[] results = new String[cvPacks.length];
        for(int i=0; i<round; i++) {
            for(CVPack cvPack : cvPacks) {
                cvPack.update(i);
            }
            
            for(int j=0; j<cvPacks.length; j++) {
                results[j] = cvPacks[j].eval(i);
            }
            
            evalHist[i] = aggCVResults(results);
            logger.info(evalHist[i]);
        }
        return evalHist;
    }
    
    /**
     * make an n-fold array of CVPack from random indices
     * @param data
     * @param nfold
     * @param params
     * @param evalMetrics
     * @return 
     */
    public static CVPack[] makeNFold(DMatrix data, int nfold, Map<String, String> params, String[] evalMetrics) {
        List<Integer> samples = genRandPermutationNums(0, (int) data.rowNum());
        int step = samples.size()/nfold;
        int[] testSlice = new int[step];
        int[] trainSlice = new int[samples.size()-step];
        int testid, trainid;
        CVPack[] cvPacks = new CVPack[nfold];
        for(int i=0; i<nfold; i++) {
            testid = 0;
            trainid = 0;
            for(int j=0; j<samples.size(); j++) {
                if(j>(i*step) && j<(i*step+step) && testid<step) {
                    testSlice[testid] = samples.get(j);
                    testid++;
                }
                else{
                    if(trainid<samples.size()-step) {
                        trainSlice[trainid] = samples.get(j);
                        trainid++;
                    }
                    else {
                        testSlice[testid] = samples.get(j);
                        testid++;
                    }
                }
            }
            
            DMatrix dtrain = data.slice(trainSlice);
            DMatrix dtest = data.slice(testSlice);
            CVPack cvPack = new CVPack(dtrain, dtest, params);
            //set eval types
            if(evalMetrics!=null) {
                for(String type : evalMetrics) {
                    cvPack.booster.setParam("eval_metric", type);
                }
            }
            cvPacks[i] = cvPack;
        }
        
        return cvPacks;
    }
    
    private static List<Integer> genRandPermutationNums(int start, int end) {
        List<Integer> samples = new ArrayList<>();
        for(int i=start; i<end; i++) {
            samples.add(i);
        }
        Collections.shuffle(samples);
        return samples;
    }
    
    /**
     * Aggregate cross-validation results.
     * @param results
     * @return 
     */
    public static String aggCVResults(String[] results) {
        Map<String, List<Float> > cvMap = new HashMap<>();
        String aggResult = results[0].split("\t")[0];
        for(String result : results) {
            String[] items = result.split("\t");
            for(int i=1; i<items.length; i++) {
                String[] tup = items[i].split(":");
                String key = tup[0];
                Float value = Float.valueOf(tup[1]);
                if(!cvMap.containsKey(key)) {
                    cvMap.put(key, new ArrayList<Float>());
                }
                cvMap.get(key).add(value);
            }
        }
        
        for(String key : cvMap.keySet()) {
            float value = 0f;
            for(Float tvalue : cvMap.get(key)) {
                value += tvalue;
            }
            value /= cvMap.get(key).size();
            aggResult += String.format("\tcv-%s:%f", key, value);
        }
        
        return aggResult;
    }
}
