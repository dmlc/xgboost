#!/bin/bash
# Simple script to test distributed version, to be deleted later.
cd xgboost4j-demo
../../dmlc-core/tracker/dmlc-submit --cluster=local --num-workers=3 java -cp target/xgboost4j-demo-0.1-jar-with-dependencies.jar ml.dmlc.xgboost4j.demo.DistTrain
cd ..
