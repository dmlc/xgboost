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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import org.dmlc.xgboost4j.DMatrix;

/**
 * class to handle evaluation dmatrix
 * @author hzx
 */
public class WatchList implements Iterable<Entry<String, DMatrix> >{
    List<Entry<String, DMatrix>> watchList = new ArrayList<>();
    
    /**
     * put eval dmatrix and it's name 
     * @param name
     * @param dmat 
     */
    public void put(String name, DMatrix dmat) {
        watchList.add(new AbstractMap.SimpleEntry<>(name, dmat));
    }
    
    public int size() {
        return watchList.size();
    }

    @Override
    public Iterator<Entry<String, DMatrix>> iterator() {
        return watchList.iterator();
    }    
}
