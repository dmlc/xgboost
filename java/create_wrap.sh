echo "build java wrapper"
cd ..
make java
echo "move native lib"
cd java
rm -f xgboost4j/src/main/resources/lib/libxgboostjavawrapper.so
mv libxgboostjavawrapper.so xgboost4j/src/main/resources/lib/
echo "complete"