#!/bin/bash
# Simple script to test distributed version, to be deleted later.
cd xgboost4j-demo
java  -XX:OnError="gdb - %p" -cp target/xgboost4j-demo-0.1-jar-with-dependencies.jar ml.dmlc.xgboost4j.demo.DistTrain 4
cd ..
