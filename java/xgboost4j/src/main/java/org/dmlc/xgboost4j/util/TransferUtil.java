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

import org.dmlc.xgboost4j.DMatrix;

/**
 *
 * @author hzx
 */
public class TransferUtil {
    /**
     * transfer DMatrix array to handle array (used for native functions)
     * @param dmatrixs
     * @return handle array for input dmatrixs
     */
    public static long[] dMatrixs2handles(DMatrix[] dmatrixs) {
        long[] handles = new long[dmatrixs.length];
        for(int i=0; i<dmatrixs.length; i++) {
            handles[i] = dmatrixs[i].getHandle();
        }
        return handles;
    }
    
    /**
     * flatten a mat to array
     * @param mat
     * @return 
     */
    public static float[] flatten(float[][] mat) {
        int size = 0;
        for (float[] array : mat) size += array.length;
        float[] result = new float[size];
        int pos = 0;
        for (float[] ar : mat) {
            System.arraycopy(ar, 0, result, pos, ar.length);
            pos += ar.length;
        }
        
        return result;
    }
}
