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

import java.io.IOException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.wrapper.XgboostJNI;

/**
 * error handle for Xgboost
 * @author hzx
 */
public class ErrorHandle {
    private static final Log logger = LogFactory.getLog(ErrorHandle.class);
    
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
     * check the return value of C API
     * @param ret return valud of xgboostJNI C API call
     * @throws org.dmlc.xgboost4j.util.XGBoostError
     */
    public static void checkCall(int ret) throws XGBoostError {
        if(ret != 0) {
            throw new XGBoostError(XgboostJNI.XGBGetLastError());
        }
    }
}
