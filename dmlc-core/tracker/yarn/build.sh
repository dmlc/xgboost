#!/bin/bash
if [ ! -d bin ]; then
    mkdir bin
fi

CPATH=`${HADOOP_HOME}/bin/hadoop classpath`
${JAVA_HOME}/bin/javac -cp $CPATH -d bin src/main/java/org/apache/hadoop/yarn/dmlc/*.java
${JAVA_HOME}/bin/jar cf dmlc-yarn.jar -C bin .
