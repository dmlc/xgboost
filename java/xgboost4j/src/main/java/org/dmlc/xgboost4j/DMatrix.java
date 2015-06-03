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

import java.io.IOException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.util.Initializer;
import org.dmlc.xgboost4j.wrapper.XgboostJNI;

/**
 * DMatrix for xgboost, similar to the python wrapper xgboost.py
 * @author hzx
 */
public class DMatrix {
    private static final Log logger = LogFactory.getLog(DMatrix.class);    
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
    
    //sparse matrix type
    public static enum SparseType {
        CSR,
        CSC;
    }
    
    /**
     *  dataPath should be svmlight format
     * @param dataPath 
     */
    public DMatrix(String dataPath) {
        handle = XgboostJNI.XGDMatrixCreateFromFile(dataPath, 1);
    }
    
    /**
     * init DMatrix withsparse matrixs
     * @param headers index to headers
     * @param indices Indices
     * @param data values
     * @param st sparse matrix type (CSR or CSC)
     */
    public DMatrix(long[] headers, long[] indices, float[] data, SparseType st) {
        if(st == SparseType.CSR) {
            handle = XgboostJNI.XGDMatrixCreateFromCSR(headers, indices, data);
        }
        else if(st == SparseType.CSC) {
            handle = XgboostJNI.XGDMatrixCreateFromCSC(headers, indices, data);
        }
        else {
            throw new UnknownError("unknow sparsetype");
        }
    }
    
   /**
     * create DMatrix content from dense matrix
     * @param data data values
     * @param nrow number of rows
     * @param ncol number of columns
     */
    public DMatrix(float[] data, int nrow, int ncol) {
        handle = XgboostJNI.XGDMatrixCreateFromMat(data, nrow, ncol, 0.0f);
    }
    
    /**
     * used for DMatrix slice
     * @param handle 
     */
    private DMatrix(long handle) {
        this.handle = handle;
    }
    
    
    
    /**
     * set label of dmatrix
     * @param labels 
     */
    public void setLabel(float[] labels) {
        XgboostJNI.XGDMatrixSetFloatInfo(handle, "label", labels);
    }
    
    /**
     * set weight of each instance
     * @param weights 
     */
    public void setWeight(float[] weights) {
        XgboostJNI.XGDMatrixSetFloatInfo(handle, "weight", weights);
    }
    
    /**
     * if specified, xgboost will start from this init margin
     * can be used to specify initial prediction to boost from
     * @param baseMargin 
     */
    public void setBaseMargin(float[] baseMargin) {
        XgboostJNI.XGDMatrixSetFloatInfo(handle, "base_margin", baseMargin);
    }
    
    /**
     * Set group sizes of DMatrix (used for ranking)
     * @param group 
     */
    public void setGroup(int[] group) {
        XgboostJNI.XGDMatrixSetGroup(handle, group);
    }
    
    private float[] getFloatInfo(String field) {
        float[] infos = XgboostJNI.XGDMatrixGetFloatInfo(handle, field);
        return infos;
    }
    
    private int[] getIntInfo(String field) {
        int[] infos = XgboostJNI.XGDMatrixGetUIntInfo(handle, field);
        return infos;
    }
    
    /**
     * get label values
     * @return 
     */
    public float[] getLabel() {
        return getFloatInfo("label");
    }
    
    /**
     * get weight of the DMatrix
     * @return 
     */
    public float[] getWeight() {
        return getFloatInfo("weight");
    }
    
    /**
     * get base margin of the DMatrix
     * @return 
     */
    public float[] getBaseMargin() {
        return getFloatInfo("base_margin");
    }
    
    /**
     * Slice the DMatrix and return a new DMatrix that only contains `rowIndex`.
     * @param rowIndex
     * @return 
     */
    public DMatrix slice(int[] rowIndex) {
        long sHandle = XgboostJNI.XGDMatrixSliceDMatrix(handle, rowIndex);
        DMatrix sMatrix = new DMatrix(sHandle);
        return sMatrix;
    }
    
    /**
     * get the row number of DMatrix
     * @return 
     */
    public long rowNum() {
        return XgboostJNI.XGDMatrixNumRow(handle);
    }
    
    /**
     * save DMatrix to filePath
     * @param filePath 
     */
    public void saveBinary(String filePath) {
        XgboostJNI.XGDMatrixSaveBinary(handle, filePath, 1);
    }
    
    public long getHandle() {
        return handle;
    }
    
    @Override
    protected void finalize() {
        delete();
    }
    
    public synchronized void delete() {
        if(handle != 0) {
            XgboostJNI.XGDMatrixFree(handle);
            handle = 0;
        }
    }
}
