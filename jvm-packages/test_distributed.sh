#!/bin/bash
# Simple script to test distributed version, to be deleted later.
cd xgboost4j-flink
flink run -c ml.dmlc.xgboost4j.flink.Test -p 4  target/xgboost4j-flink-0.1-jar-with-dependencies.jar
cd ..
