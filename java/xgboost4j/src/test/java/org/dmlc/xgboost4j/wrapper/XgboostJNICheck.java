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
package org.dmlc.xgboost4j.wrapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.dmlc.xgboost4j.util.Initializer;
import org.junit.Test;
import org.junit.BeforeClass;

/**
 * check the wrapper functions
 * @author hzx
 */
public class XgboostJNICheck {
    @BeforeClass
    public static void init() throws IOException {
        Initializer.InitXgboost();
    }
    
    @Test
    public void testXGDMatrix() {
        //create from file
        long handle = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_test.txt", 0);
        XgboostJNI.XGDMatrixFree(handle);
        
        //create from CSR
        /**
         * sparse matrix
         * 1 0 2 3 0
         * 4 0 2 3 5
         * 3 1 2 5 0
         */
        float[] data = new float[] {1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5};
        long[] colIndex = new long[] {0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3};
        long[] rowHeaders = new long[] {0, 3, 7, 11};
        handle = XgboostJNI.XGDMatrixCreateFromCSR(rowHeaders, colIndex, data);
        XgboostJNI.XGDMatrixFree(handle);
        
        //create from mat
        //create DMatrix from 10*10 dense matrix
        int nrow = 10;
        int ncol = 10;
        float[] data0 = new float[nrow*ncol];
        //put random nums
        Random random = new Random();
        for(int i=0; i<nrow*ncol; i++) {
            data0[i] = random.nextFloat();
        }
        
        handle = XgboostJNI.XGDMatrixCreateFromMat(data0, nrow, ncol, 0.0f);
        XgboostJNI.XGDMatrixFree(handle);
        
        //get slice of mat
        long handle0 = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_test.txt", 0);
        long handle1 = XgboostJNI.XGDMatrixSliceDMatrix(handle0, new int[] {0, 1, 3, 5, 6});
        
        XgboostJNI.XGDMatrixFree(handle0);
        XgboostJNI.XGDMatrixFree(handle1);
        
        //save dmatrix
        String matPath = "./tmp/dmat.bin";
        handle = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_test.txt", 0);
        int silent = 0;
        XgboostJNI.XGDMatrixSaveBinary(handle, matPath, silent);
        XgboostJNI.XGDMatrixFree(handle);
        
        //set float info
        float[] label0 = new float[10];
        for(int i=0; i<nrow; i++) {
            label0[i] = random.nextFloat();
        }
        handle = XgboostJNI.XGDMatrixCreateFromMat(data0, nrow, ncol, 0.0f);
        XgboostJNI.XGDMatrixSetFloatInfo(handle, "label", label0);
        //get float info
        float[] label1 = XgboostJNI.XGDMatrixGetFloatInfo(handle, "label");
        System.out.println(Arrays.toString(label0));
        System.out.println(Arrays.toString(label1));
        
        
        
//        //set uint info
//        int[] info = new int[] {0,1,2,3,4,5,6,7,8, 9};
//        XgboostJNI.XGDMatrixSetUIntInfo(handle, "", info);
        
        //get row num
        long rows = XgboostJNI.XGDMatrixNumRow(handle);
        System.out.println("rows: " + rows);
        
        
        XgboostJNI.XGDMatrixFree(handle);
    }
    
    @Test
    public void testXGBooster() {
        
        //test initialize booster
        long handle0 = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_train.txt", 0);
        long handle1 = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_valid.txt", 0);
        
        long[] dmats = new long[] {handle0, handle1};
        long bst_handle = XgboostJNI.XGBoosterCreate(dmats);
        
        //params
        Map<String,String> params = new HashMap<>();
        params.put("seed", "0");
        params.put("eta", "0.3");
        params.put("max_depth", "6");
        params.put("silent", "1");
        params.put("nthread", "6");
        params.put("num_class", "9");
        params.put("objective", "multi:softprob");
        params.put("eval_metric", "mlogloss");
         
        for(Map.Entry<String,String> entry : params.entrySet()) {
            XgboostJNI.XGBoosterSetParam(bst_handle, entry.getKey(), entry.getValue());
        }
        
        //train one iter
        System.out.println("begin to train");
        String[] evnames = new String[] {"train", "valid"};
        int iters = 3;
        for(int iter=0; iter<iters; iter++) {
            XgboostJNI.XGBoosterUpdateOneIter(bst_handle, 0, handle0);
            String evalInfo = XgboostJNI.XGBoosterEvalOneIter(bst_handle, 0, dmats, evnames);
            System.out.println(evalInfo);
        }
        
        //test predict
        long handle2 = XgboostJNI.XGDMatrixCreateFromFile("./tmp/final_test.txt", 0);
        float[] predicts = XgboostJNI.XGBoosterPredict(bst_handle, handle2, 0, 0);
        
        System.out.println(predicts.length);
        
        //save model
        String modelPath = "./tmp/test_model";
        XgboostJNI.XGBoosterSaveModel(bst_handle, modelPath);
        
        //dump model
        String[] modelInfo = XgboostJNI.XGBoosterDumpModel(bst_handle, "", 1);
        System.out.println(modelInfo.length);
        System.out.println(modelInfo[0]);
        
         System.out.println("**************************************");
        
        //free
        XgboostJNI.XGDMatrixFree(handle0);
        XgboostJNI.XGDMatrixFree(handle1);
        XgboostJNI.XGDMatrixFree(handle2);
        XgboostJNI.XGBoosterFree(bst_handle);
        
    }
}
