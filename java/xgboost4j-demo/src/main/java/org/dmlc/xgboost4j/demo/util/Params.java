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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.AbstractMap;


/**
 * a util class for handle params
 * @author hzx
 */
public class Params implements Iterable<Entry<String, Object>>{
    List<Entry<String, Object>> params = new ArrayList<>();
    
    /**
     * put param key-value pair
     * @param key
     * @param value 
     */
    public void put(String key, Object value) {
        params.add(new AbstractMap.SimpleEntry<>(key, value));
    }
    
    @Override
    public String toString(){ 
        String paramsInfo = "";
        for(Entry<String, Object> param : params) {
            paramsInfo += param.getKey() + ":" + param.getValue() + "\n";
        }
        return paramsInfo;
    }

    @Override
    public Iterator<Entry<String, Object>> iterator() {
        return params.iterator();
    }
}
