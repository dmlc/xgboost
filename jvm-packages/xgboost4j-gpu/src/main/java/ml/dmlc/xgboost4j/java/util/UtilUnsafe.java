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
package ml.dmlc.xgboost4j.java.util;

import java.lang.reflect.Field;

import sun.misc.Unsafe;

/**
 * Simple class to obtain access to the {@link Unsafe} object. Use responsibly :)
 */
public final class UtilUnsafe {

  static Unsafe UNSAFE = getUnsafe();

  private UtilUnsafe() {
  } // dummy private constructor

  private static Unsafe getUnsafe() {
    // Not on bootclasspath
    if (UtilUnsafe.class.getClassLoader() == null) {
      return Unsafe.getUnsafe();
    }
    try {
      final Field fld = Unsafe.class.getDeclaredField("theUnsafe");
      fld.setAccessible(true);
      return (Unsafe) fld.get(UtilUnsafe.class);
    } catch (Exception e) {
      throw new RuntimeException("Could not obtain access to sun.misc.Unsafe", e);
    }
  }

}
