echo "build java wrapper"
cd ..
make
cd java
make java
echo "move native lib"
rm -f xgboost4j/src/main/resources/lib/libxgboostjavawrapper.so
mv libxgboostjavawrapper.so xgboost4j/src/main/resources/lib/
echo "complete"