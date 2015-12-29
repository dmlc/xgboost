#!/usr/bin/env sh
echo "build java wrapper"

# cd to script's directory
pushd `dirname $0` > /dev/null

#settings according to os
dl="so"
omp="0"
if [ $(uname) == "Darwin" ]; then
  export JAVA_HOME=$(/usr/libexec/java_home)
  dl="dylib"
  omp="1"
fi

cd ..
make java no_omp=${omp}
cd java
echo "move native lib"

libPath="xgboost4j/src/main/resources/lib"
if [ ! -d "$libPath" ]; then
  mkdir -p "$libPath"
fi

rm -f xgboost4j/src/main/resources/lib/libxgboost4j.${dl}
mv libxgboost4j.so xgboost4j/src/main/resources/lib/libxgboost4j.${dl}

popd > /dev/null
echo "complete"
