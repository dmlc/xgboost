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

package ml.dmlc.xgboost4j.scala.rabit.util

import java.nio.{ByteOrder, ByteBuffer}
import akka.util.ByteString

private[rabit] object RabitTrackerHelpers {
  implicit class ByteStringHelplers(bs: ByteString) {
    // Java by default uses big endian. Enforce native endian so that
    // the byte order is consistent with the workers.
    def asNativeOrderByteBuffer: ByteBuffer = {
      bs.asByteBuffer.order(ByteOrder.nativeOrder())
    }
  }

  implicit class ByteBufferHelpers(buf: ByteBuffer) {
    def getString: String = {
      val len = buf.getInt()
      val stringBuffer = ByteBuffer.allocate(len).order(ByteOrder.nativeOrder())
      buf.get(stringBuffer.array(), 0, len)
      new String(stringBuffer.array(), "utf-8")
    }
  }
}
