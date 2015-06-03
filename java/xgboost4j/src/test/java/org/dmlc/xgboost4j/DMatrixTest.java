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

import java.util.Arrays;
import java.util.Random;
import junit.framework.TestCase;

import org.junit.Test;


/**
 *
 * @author hzx
 */
public class DMatrixTest {
    @Test
    public void testCreateFromFile() {
         //create DMatrix from file
        DMatrix dmat = new DMatrix("./tmp/final_test.txt");
        //get label
        float[] labels = dmat.getLabel();
        //check length
        TestCase.assertTrue(dmat.rowNum()==labels.length);
        //set weights
        float[] weights = Arrays.copyOf(labels, labels.length);
        dmat.setWeight(weights);
        float[] dweights = dmat.getWeight();
        TestCase.assertTrue(Arrays.equals(weights, dweights));
        //release
        dmat.delete();
    }
    
    @Test
    public void testCreateFromCSR() {
        //create Matrix from csr format sparse Matrix and labels
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
        //check row num
        System.out.println(dmat1.rowNum());
        TestCase.assertTrue(dmat1.rowNum()==3);
        //set label
        float[] label1 = new float[] {1, 0, 1};
        dmat1.setLabel(label1);
        float[] label2  = dmat1.getLabel();
        TestCase.assertTrue(Arrays.equals(label1, label2));
        
        dmat1.delete();
    }
    
    @Test
    public void testCreateFromCSC() {
        //create dmatrix from csc format sparse matrix and labels
        /**
         * sparse matrix
         * 1 0 2 3 0
         * 4 0 2 3 5
         * 3 1 2 5 0
         */
        float[] data = new float[] {1, 4, 3, 1, 2, 2, 2, 3, 3, 5, 5};
        long[] colHeaders = new long[] {0, 3, 4, 7, 10, 11};
        long[] rowIndex = new long[] {0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 1};
        DMatrix dmat1 = new DMatrix(colHeaders, rowIndex, data, DMatrix.SparseType.CSC);
         //check row num
        System.out.println(dmat1.rowNum());
        TestCase.assertTrue(dmat1.rowNum()==3);
        //set label
        float[] label1 = new float[] {1, 0, 1};
        dmat1.setLabel(label1);
        float[] label2  = dmat1.getLabel();
        TestCase.assertTrue(Arrays.equals(label1, label2));
    }
    
    @Test
    public void testCreateFromDenseMatrix() {
         //create DMatrix from 10*5 dense matrix
        int nrow = 10;
        int ncol = 5;
        float[] data0 = new float[nrow*ncol];
        //put random nums
        Random random = new Random();
        for(int i=0; i<nrow*ncol; i++) {
            data0[i] = random.nextFloat();
        }
        
        //create label
        float[] label0 = new float[nrow];
        for(int i=0; i<nrow; i++) {
            label0[i] = random.nextFloat();
        }
        
        DMatrix dmat0 = new DMatrix(data0, nrow, ncol);
        dmat0.setLabel(label0);
        
        //check
        TestCase.assertTrue(dmat0.rowNum()==10);
        TestCase.assertTrue(dmat0.getLabel().length==10);
        
        //set weights for each instance
        float[] weights = new float[nrow];
        for(int i=0; i<nrow; i++) {
            weights[i] = random.nextFloat();
        }
        dmat0.setWeight(weights);
        
        TestCase.assertTrue(Arrays.equals(weights, dmat0.getWeight()));
        
        dmat0.delete();
    }
}
