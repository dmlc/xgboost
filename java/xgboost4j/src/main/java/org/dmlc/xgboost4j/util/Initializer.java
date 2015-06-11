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
import java.lang.reflect.Field;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * class to load native library
 * @author hzx
 */
public class Initializer {
    private static final Log logger = LogFactory.getLog(Initializer.class);
    
    static boolean initialized = false;
    public static final String nativePath = "./lib";
    public static final String nativeResourcePath = "/lib/";
    public static final String[] libNames = new String[] {"xgboostjavawrapper"};
    
    public static synchronized void InitXgboost() throws IOException {
        if(initialized == false) {
            for(String libName: libNames) {
                smartLoad(libName);
            }
            initialized = true;
        }
    }
    
    /**
     * load native library, this method will first try to load library from java.library.path, then try to load library in jar package.
     * @param libName
     * @throws IOException 
     */
    private static void smartLoad(String libName) throws IOException {
        addNativeDir(nativePath);
        try {
             System.loadLibrary(libName);
         } 
         catch (UnsatisfiedLinkError e) {
             try {
                 NativeUtils.loadLibraryFromJar(nativeResourcePath + System.mapLibraryName(libName));
             }
             catch (IOException e1) {
                 throw e1;
             }
         }
    }
    
    /**
     * add libPath to java.library.path, then native library in libPath would be load properly
     * @param libPath
     * @throws IOException 
     */
    public static void addNativeDir(String libPath) throws IOException {
        try {
            Field field = ClassLoader.class.getDeclaredField("usr_paths");
            field.setAccessible(true);
            String[] paths = (String[]) field.get(null);
            for (String path : paths) {
                if (libPath.equals(path)) {
                    return;
                }
            }
            String[] tmp = new String[paths.length+1];
            System.arraycopy(paths,0,tmp,0,paths.length);
            tmp[paths.length] = libPath;
            field.set(null, tmp);
        } catch (IllegalAccessException  e) {
            logger.error(e.getMessage());
            throw new IOException("Failed to get permissions to set library path"); 
        } catch (NoSuchFieldException e) {
            logger.error(e.getMessage());
            throw new IOException("Failed to get field handle to set library path"); 
        }
    }
}
