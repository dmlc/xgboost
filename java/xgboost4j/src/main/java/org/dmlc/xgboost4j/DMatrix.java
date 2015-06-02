/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
    
    /**
     *  dataPath should be svmlight format
     * @param dataPath 
     */
    public DMatrix(String dataPath) {
        handle = XgboostJNI.XGDMatrixCreateFromFile(dataPath, 1);
    }
    
    /**
     * init DMatrix with CSR sparse matrixs
     * @param rowHeaders index to row headers
     * @param colIndices colIndices
     * @param data values
     */
    public DMatrix(long[] rowHeaders, long[] colIndices, float[] data) {
        handle = XgboostJNI.XGDMatrixCreateFromCSR(rowHeaders, colIndices, data);
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
