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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;

/**
 * util class for loading data
 * @author hzx
 */
public class DataLoader {
    public static class DenseData {
        public float[] labels;
        public float[] data;
        public int nrow;
        public int ncol;
    }
    
    public static class CSRSparseData {
        public float[] labels;
        public float[] data;
        public long[] rowHeaders;
        public int[] colIndex;
    }
    
    public static DenseData loadCSVFile(String filePath) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        DenseData denseData = new DenseData();
        
        File f = new File(filePath);
        FileInputStream in = new FileInputStream(f);
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, "UTF-8"));
        
        denseData.nrow = 0;
        denseData.ncol = -1;
        String line;
        List<Float> tlabels = new ArrayList<>();
        List<Float> tdata = new ArrayList<>();
        
        while((line=reader.readLine()) != null) {
            String[] items = line.trim().split(",");
            if(items.length==0) {
                continue;
            }
            denseData.nrow++;
            if(denseData.ncol == -1) {
                denseData.ncol = items.length - 1;
            }
            
            tlabels.add(Float.valueOf(items[items.length-1]));
            for(int i=0; i<items.length-1; i++) {
                tdata.add(Float.valueOf(items[i]));
            }
        }
        
        reader.close();
        in.close();
        
        denseData.labels = ArrayUtils.toPrimitive(tlabels.toArray(new Float[tlabels.size()]));
        denseData.data = ArrayUtils.toPrimitive(tdata.toArray(new Float[tdata.size()]));
        
        return denseData;
    }
    
    public static CSRSparseData loadSVMFile(String filePath) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        CSRSparseData spData = new CSRSparseData();
        
        List<Float> tlabels = new ArrayList<>();
        List<Float> tdata = new ArrayList<>();
        List<Long> theaders = new ArrayList<>();
        List<Integer> tindex = new ArrayList<>();
        
        File f = new File(filePath);
        FileInputStream in = new FileInputStream(f);
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, "UTF-8"));
        
        String line;
        long rowheader = 0;
        theaders.add(rowheader);
        while((line=reader.readLine()) != null) {
            String[] items = line.trim().split(" ");
            if(items.length==0) {
                continue;
            }
            
            rowheader += items.length - 1;
            theaders.add(rowheader);
            tlabels.add(Float.valueOf(items[0]));
            
            for(int i=1; i<items.length; i++) {
                String[] tup = items[i].split(":");
                assert tup.length == 2;
                
                tdata.add(Float.valueOf(tup[1]));
                tindex.add(Integer.valueOf(tup[0]));
            }
        }
        
        spData.labels = ArrayUtils.toPrimitive(tlabels.toArray(new Float[tlabels.size()]));
        spData.data = ArrayUtils.toPrimitive(tdata.toArray(new Float[tdata.size()]));
        spData.colIndex = ArrayUtils.toPrimitive(tindex.toArray(new Integer[tindex.size()]));
        spData.rowHeaders = ArrayUtils.toPrimitive(theaders.toArray(new Long[theaders.size()]));
        
        return spData;
    }
}
