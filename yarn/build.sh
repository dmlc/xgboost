#!/bin/bash
if [ ! -d bin ]; then
    mkdir bin
fi

CPATH=`${HADOOP_HOME}/bin/hadoop classpath`
javac -cp $CPATH -d bin src/org/apache/hadoop/yarn/rabit/*
jar cf rabit-yarn.jar -C bin . 
