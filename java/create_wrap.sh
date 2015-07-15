echo "build java wrapper"
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

echo "complete"
