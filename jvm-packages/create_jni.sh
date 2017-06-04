#!/usr/bin/env bash

set -e -x

echo "build java wrapper"

# cd to script's directory
pushd `dirname $0` > /dev/null

#settings according to os
dl="so"
USE_OMP=ON

if [ $(uname) == "Darwin" ]; then
  export JAVA_HOME=$(/usr/libexec/java_home)
  dl="dylib"
  #change this to 0 if your compiler support openmp
  USE_OMP=OFF
fi

cd ..
mkdir -p build
cd build
cmake .. -DJVM_BINDINGS:BOOL=ON -DUSE_OPENMP:BOOL=${USE_OMP}
make
cd ../jvm-packages
echo "move native lib"

libPath="xgboost4j/src/main/resources/lib"
if [ ! -d "$libPath" ]; then
  mkdir -p "$libPath"
fi

rm -f xgboost4j/src/main/resources/lib/libxgboost4j.${dl}
cp ../lib/libxgboost4j.${dl} xgboost4j/src/main/resources/lib/libxgboost4j.${dl}
# copy python to native resources
cp ../dmlc-core/tracker/dmlc_tracker/tracker.py xgboost4j/src/main/resources/tracker.py
# copy test data files
mkdir -p xgboost4j-spark/src/test/resources/
cd ../demo/regression
python mapfeat.py
python mknfold.py machine.txt 1
cd -
cp ../demo/regression/machine.txt.t* xgboost4j-spark/src/test/resources/
cp ../demo/data/agaricus.* xgboost4j-spark/src/test/resources/
popd > /dev/null
echo "complete"
