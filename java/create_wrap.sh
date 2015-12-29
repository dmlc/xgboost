echo "build java wrapper"
dylib="so"
dis_omp=0
if [ $(uname) == "Darwin" ]; then
  export JAVA_HOME=$(/usr/libexec/java_home)
  dylib="dylib"
  #change this to 0 if your compiler support openmp
  dis_omp=1
fi
cd ..
make java no_omp=${dis_omp}
cd java
echo "move native lib"

libPath="xgboost4j/src/main/resources/lib"
if [ ! -d "$libPath" ]; then
  mkdir -p "$libPath"
fi

rm -f xgboost4j/src/main/resources/lib/libxgboost4j.${dylib}
mv libxgboost4j.so xgboost4j/src/main/resources/lib/libxgboost4j.${dylib}

echo "complete"
