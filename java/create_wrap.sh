#!/usr/bin/env sh

echo "build java wrapper"

# cd to script's directory
pushd `dirname $0` > /dev/null

cd ..
make java
cd java
echo "move native lib"

libPath="xgboost4j/src/main/resources/lib"
if [ ! -d "$libPath" ]; then
  mkdir -p "$libPath"
fi

rm -f xgboost4j/src/main/resources/lib/libxgboostjavawrapper.so
mv libxgboostjavawrapper.so xgboost4j/src/main/resources/lib/

popd > /dev/null
echo "complete"
