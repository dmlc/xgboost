#!/bin/bash
if [ -z "$HADOOP_PREFIX" ]; then
    echo "cannot found $HADOOP_PREFIX in the environment variable, please set it properly"
    exit 1
fi
CPATH=`${HADOOP_PREFIX}/bin/hadoop classpath`
javac -cp $CPATH -d bin src/org/apache/hadoop/yarn/rabit/*
jar cf rabit-yarn.jar -C bin . 
