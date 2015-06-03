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
import java.util.Arrays;
import java.util.Random;

/**
 * DMatrixCheck
 * @author hzx
 */
public class DMatrixCheck {
    public static void main(String[] args) throws IOException {
        //create DMatrix from file
        DMatrix dmat = new DMatrix("./tmp/final_test.txt");
        //get label
        float[] labels = dmat.getLabel();
        System.out.println(labels.length);
        System.out.println(Arrays.toString(labels));
        //get rownum
        System.out.println(dmat.rowNum());
        //set weights
        float[] weights = Arrays.copyOf(labels, labels.length);
        dmat.setWeight(weights);
        float[] dweights = dmat.getWeight();
        System.out.println(dweights.length);
        System.out.println(Arrays.toString(weights));
        
        //get slice
        DMatrix smat  = dmat.slice(new int[] {1,2,3,7,9,11,13,17,23,85,79,3000});
        System.out.println(smat.rowNum());
        System.out.println(Arrays.toString(smat.getLabel()));
        
        //create DMatrix from 10*10 dense matrix
        int nrow = 10;
        int ncol = 10;
        float[] data0 = new float[nrow*ncol];
        //put random nums
        Random random = new Random();
        for(int i=0; i<nrow*ncol; i++) {
            data0[i] = random.nextFloat();
        }
        
        float[] label0 = new float[nrow];
        for(int i=0; i<nrow; i++) {
            label0[i] = random.nextFloat();
        }
        
        DMatrix dmat0 = new DMatrix(data0, nrow, ncol);
        dmat0.setLabel(label0);
        
        //get row and labels
        System.out.println(dmat0.rowNum());
        System.out.println(Arrays.toString(dmat0.getLabel()));
        
        //create Matrix from csr format sparse Matrix
        /**
         * sparse matrix
         * 1 0 2 3 0
         * 4 0 2 3 5
         * 3 1 2 5 0
         */
        float[] data = new float[] {1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5};
        long[] colIndex = new long[] {0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3};
        long[] rowHeaders = new long[] {0, 3, 7, 11};
        DMatrix dmat1 = new DMatrix(rowHeaders, colIndex, data, DMatrix.SparseType.CSR);
        float[] label1 = new float[] {1, 0, 1};
        dmat1.setLabel(label1);
        
        System.out.println(dmat1.rowNum());
        System.out.println(Arrays.toString(dmat1.getLabel()));
        
         //release memory
        dmat.delete();
        dmat0.delete();
        dmat1.delete();
    }
}
