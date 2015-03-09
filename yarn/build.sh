#!/bin/bash
CPATH=`${HADOOP_PREFIX}/bin/hadoop classpath`
javac -cp $CPATH -d bin src/org/apache/hadoop/yarn/rabit/*
jar cf rabit-yarn.jar -C bin . 
