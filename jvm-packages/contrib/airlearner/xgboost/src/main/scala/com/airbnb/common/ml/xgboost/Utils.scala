package com.airbnb.common.ml.xgboost

import java.io.File
import java.util.UUID


object Utils {

  def delFile(file: String): Unit = {
    if (!file.isEmpty) {
      new File(file).delete()
    }
  }

  def generateTmpFile(tmpFolder: String): String = {
    s"$tmpFolder${UUID.randomUUID()}"
  }

  def getHDFSDataFile(dataDir: String, id: String, dataFile: String): String = {
    s"$dataDir$id$dataFile"
  }

}
