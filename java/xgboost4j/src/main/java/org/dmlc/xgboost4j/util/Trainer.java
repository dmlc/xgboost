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
import java.util.Map.Entry;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IObjective;


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
     * @param watchs a group of items to be evaluated during training, this allows user to watch performance on the validation set.
     * @param obj customized objective (set to null if not used)
     * @param eval customized evaluation (set to null if not used)
     * @return trained booster
     */
    public static Booster train(Iterable<Entry<String, Object>> params, DMatrix dtrain, int round, 
            Iterable<Entry<String, DMatrix>> watchs, IObjective obj, IEvaluation eval) throws XGBoostError {
        
        //collect eval matrixs
        String[] evalNames;
        DMatrix[] evalMats;
        List<String> names = new ArrayList<>();
        List<DMatrix> mats = new ArrayList<>();
        
        for(Entry<String, DMatrix> evalEntry : watchs) {
            names.add(evalEntry.getKey());
            mats.add(evalEntry.getValue());
        }
        
        evalNames = names.toArray(new String[names.size()]);
        evalMats = mats.toArray(new DMatrix[mats.size()]);
        
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
        
        //begin to train
        for(int iter=0; iter<round; iter++) {
            if(obj != null) {
                booster.update(dtrain, iter, obj);
            } else {
                booster.update(dtrain, iter);
            }
            
            //evaluation
            if(evalMats!=null && evalMats.length>0) {
                String evalInfo;
                if(eval != null) {
                    evalInfo = booster.evalSet(evalMats, evalNames, iter, eval);
                }
                else {
                    evalInfo = booster.evalSet(evalMats, evalNames, iter);
                }
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
     * @param obj customized objective (set to null if not used)
     * @param eval customized evaluation (set to null if not used)
     * @return evaluation history
     */
    public static String[] crossValiation(Iterable<Entry<String, Object>> params, DMatrix data, int round, int nfold, String[] metrics, IObjective obj, IEvaluation eval) throws XGBoostError {
        CVPack[] cvPacks = makeNFold(data, nfold, params, metrics);
        String[] evalHist = new String[round];
        String[] results = new String[cvPacks.length];
        for(int i=0; i<round; i++) {
            for(CVPack cvPack : cvPacks) {
                if(obj != null) {
                    cvPack.update(i, obj);
                } 
                else {
                    cvPack.update(i);
                }
            }
            
            for(int j=0; j<cvPacks.length; j++) {
                if(eval != null) {
                    results[j] = cvPacks[j].eval(i, eval);
                }
                else {
                    results[j] = cvPacks[j].eval(i);
                }
            }
            
            evalHist[i] = aggCVResults(results);
            logger.info(evalHist[i]);
        }
        return evalHist;
    }
    
    /**
     * make an n-fold array of CVPack from random indices
     * @param data original data
     * @param nfold num of folds
     * @param params booster parameters
     * @param evalMetrics Evaluation metrics
     * @return CV package array
     */
    public static CVPack[] makeNFold(DMatrix data, int nfold, Iterable<Entry<String, Object>> params, String[] evalMetrics) throws XGBoostError {
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
     * @param results eval info from each data sample
     * @return cross-validation eval info
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
