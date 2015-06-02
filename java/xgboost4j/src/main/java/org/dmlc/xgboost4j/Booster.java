/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.dmlc.xgboost4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.dmlc.xgboost4j.util.Initializer;
import org.dmlc.xgboost4j.util.TransferUtil;
import org.dmlc.xgboost4j.wrapper.XgboostJNI;


/**
 * Booster for xgboost, similar to the python wrapper xgboost.py
 * but custom obj function and eval function not supported at present.
 * @author hzx
 */
public final class Booster {
    private static final Log logger = LogFactory.getLog(Booster.class);
    
    long handle = 0;
    
    //load native library
    static {
        try {
            Initializer.InitXgboost();
        } catch (IOException ex) {
            logger.error("load native library failed.");
            logger.error(ex);
        }
    }
    
    public Booster(Map<String,String> params, DMatrix[] dMatrixs) {
        init(dMatrixs);
        setParam("seed","0");
        setParams(params);
    }
    
    /**
     * load model from modelPath
     * @param params
     * @param modelPath 
     */
    public Booster(Map<String,String> params, String modelPath) {
        handle = XgboostJNI.XGBoosterCreate(new long[] {});
        loadModel(modelPath);
        setParam("seed","0");
        setParams(params);
    }
    
    private void init(DMatrix[] dMatrixs) {
        long[] handles = null;
        if(dMatrixs != null) {
            handles = TransferUtil.dMatrixs2handles(dMatrixs);
        }
        handle = XgboostJNI.XGBoosterCreate(handles);
    }
    
    public final void setParam(String key, String value) {
        XgboostJNI.XGBoosterSetParam(handle, key, value);
    }
    
    public void setParams(Map<String, String> params) {
        if(params!=null) {
            for(Map.Entry<String, String> entry : params.entrySet()) {
                setParam(entry.getKey(), entry.getValue());
            }
        }
    }
    
    /**
     * Update (one iteration)
     * @param dtrain training data
     * @param iter current iteration number
     */
    public void update(DMatrix dtrain, int iter) {
        XgboostJNI.XGBoosterUpdateOneIter(handle, iter, dtrain.getHandle());
    }
    
    /**
     * evaluate with given dmatrixs.
     * @param evalMatrixs dmatrixs for evaluation
     * @param evalNames name for eval dmatrixs, used for check results
     * @param iter current eval iteration
     * @return 
     */
    public String evalSet(DMatrix[] evalMatrixs, String[] evalNames,  int iter) {
        long[] handles = TransferUtil.dMatrixs2handles(evalMatrixs);
        String evalInfo = XgboostJNI.XGBoosterEvalOneIter(handle, iter, handles, evalNames);
        return evalInfo;
    }
    
     /**
     * evaluate with given dmatrixs.
     * @param dmats 
     * @param evalNames name for eval dmatrixs, used for check results
     * @param iter current eval iteration
     * @return 
     */
    public String evalSet(long[] dmats, String[] evalNames,  int iter) {
        String evalInfo = XgboostJNI.XGBoosterEvalOneIter(handle, iter, dmats, evalNames);
        return evalInfo;
    }
    
    
    /**
     * evaluate with given dmatrix, similar to evalSet
     * @param evalMat
     * @param evalName
     * @param iter
     * @return 
     */
    public String eval(DMatrix evalMat, String evalName, int iter) {
        DMatrix[] evalMats = new DMatrix[] {evalMat};
        String[] evalNames = new String[] {evalName};
        return  evalSet(evalMats, evalNames, iter);
    }
    
    /**
     * base function for Predict
     * @param data
     * @param outPutMargin
     * @param treeLimit
     * @param predLeaf
     * @return 
     */
    private float[][] pred(DMatrix data, boolean outPutMargin, long treeLimit, boolean predLeaf) {
        int optionMask = 0;
        if(outPutMargin) {
            optionMask = 1;
        }
        if(predLeaf) {
            optionMask = 2;
        }
        float[] rawPredicts = XgboostJNI.XGBoosterPredict(handle, data.getHandle(), optionMask, treeLimit);
        int row = (int) data.rowNum();
        int col = (int) rawPredicts.length/row;
        float[][] predicts = new float[row][col];
        int r,c;
        for(int i=0; i< rawPredicts.length; i++) {
            r = i/col;
            c = i%col;
            predicts[r][c] = rawPredicts[i];
        }
        return predicts;
    } 
    
    /**
     * Predict with data
     * @param data dmatrix storing the input
     * @return 
     */
    public float[][] predict(DMatrix data) {
        return pred(data, false, 0, false);
    }
    
    /**
     * Predict with data
     * @param data dmatrix storing the input
     * @param outPutMargin Whether to output the raw untransformed margin value.
     * @return 
     */
    public float[][] predict(DMatrix data, boolean outPutMargin) {
        return pred(data, outPutMargin, 0, false);
    }
    
    /**
     * Predict with data
     * @param data dmatrix storing the input
     * @param outPutMargin Whether to output the raw untransformed margin value.
     * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
     * @return 
     */
    public float[][] predict(DMatrix data, boolean outPutMargin, long treeLimit) {
        return pred(data, outPutMargin, treeLimit, false);
    }
    
    /**
     * Predict with data 
     * @param data dmatrix storing the input
     * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
     * @param predLeaf When this option is on, the output will be a matrix of (nsample, ntrees), nsample = data.numRow
            with each record indicating the predicted leaf index of each sample in each tree.
            Note that the leaf index of a tree is unique per tree, so you may find leaf 1
            in both tree 1 and tree 0.
     * @return 
     */
    public float[][] predict(DMatrix data , long treeLimit, boolean predLeaf) {
        return pred(data, false, treeLimit, predLeaf);
    }
    
    /**
     * save model to modelPath
     * @param modelPath 
     */
    public void saveModel(String modelPath) {
        XgboostJNI.XGBoosterSaveModel(handle, modelPath);
    }
    
    private void loadModel(String modelPath) {
        XgboostJNI.XGBoosterLoadModel(handle, modelPath);
    }
    
    /**
     * get the dump of the model as a string array
     * @param withStats
     * @return 
     */
    public String[] getDumpInfo(boolean withStats) {
        int statsFlag = 0;
        if(withStats) {
            statsFlag = 1;
        }
        String[] modelInfos = XgboostJNI.XGBoosterDumpModel(handle, "", statsFlag);
        return modelInfos;
    }
    
    /**
     * Dump model into a text file.
     * @param modelPath
     * @param withStats bool (optional)
            Controls whether the split statistics are output.
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     * @throws IOException 
     */
    public void dumpModel(String modelPath, boolean withStats) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        File tf = new File(modelPath);
        FileOutputStream out = new FileOutputStream(tf);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, "UTF-8"));
        String[] modelInfos = getDumpInfo(withStats);
        
        for(int i=0; i<modelInfos.length; i++) {
            writer.write("booster [" + i +"]:\n");
            writer.write(modelInfos[i]);
        }
        
        writer.close();
        out.close();
    }
    
    /**
     * get importance of each feature
     * @return Map key: feature index, value: feature importance score
     */
    public Map<String, Integer> getFeatureScore() {
        String[] modelInfos = getDumpInfo(false);
        Map<String, Integer> featureMap = new HashMap<>();
        for(String tree : modelInfos) {
            for(String node : tree.split("\n")) {
                String[] array = node.split("\\[");
                if(array.length == 1) {
                    continue;
                }
                String fid = array[1].split("\\]")[0];
                fid = fid.split("<")[0];
                if(featureMap.containsKey(fid)) {
                    featureMap.put(fid, 1 + featureMap.get(fid));
                }
                else {
                    featureMap.put(fid, 1);
                }
            }
        }
        return featureMap;
    }
    
    @Override
    protected void finalize() {
        delete();
    }
    
    public synchronized void delete() {
        if(handle != 0l) {
            XgboostJNI.XGBoosterFree(handle);
            handle=0;
        }
    }
}
