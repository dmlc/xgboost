#!/bin/bash
# Fail on error
set -e

# Redefine only if not defined by environment
CC=${CC:-gcc-5}
CXX=${CXX:-g++-5}

git submodule init
git submodule update --recursive
make clean
find . -name "*.so" -delete
find . -name "*.o" -delete
find . -name "*.a" -delete
rm -rf build
mkdir build
cd build
CC=$CC CXX=$CXX cmake .. -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY=../cub -DCUDA_NVCC_FLAGS="--expt-extended-lambda -arch=sm_30"
make -j40
cd ..
make jvm -j40
$CXX -std=c++14 -Wall -Wno-unknown-pragmas -Iinclude   -Idmlc-core/include -Irabit/include -O3 -funroll-loops -msse2 -fPIC -fopenmp -I/usr/lib/jvm/java-8-oracle//include -I./java -I/usr/lib/jvm/java-8-oracle//include/linux -shared -o jvm-packages/lib/libxgboost4j.so jvm-packages/xgboost4j/src/native/xgboost4j.cpp build/logging.o build/learner.o build/common/common.o build/common/hist_util.o build/metric/metric.o build/metric/rank_metric.o build/metric/elementwise_metric.o build/metric/multiclass_metric.o build/objective/multiclass_obj.o build/objective/objective.o build/objective/rank_obj.o build/objective/regression_obj.o build/data/sparse_page_dmatrix.o build/data/sparse_page_source.o build/data/sparse_page_writer.o build/data/simple_csr_source.o build/data/data.o build/data/sparse_page_raw_format.o build/data/simple_dmatrix.o build/tree/updater_prune.o build/tree/tree_updater.o build/tree/updater_refresh.o build/tree/updater_sync.o build/tree/updater_skmaker.o build/tree/updater_colmaker.o build/tree/updater_histmaker.o build/tree/tree_model.o build/tree/updater_fast_hist.o build/gbm/gbtree.o build/gbm/gblinear.o build/gbm/gbm.o build/c_api/c_api.o build/CMakeFiles/libxgboost.dir/plugin/updater_gpu/src/updater_gpu.cc.o build/c_api/c_api_error.o dmlc-core/libdmlc.a rabit/lib/librabit.a ./build/libupdater_gpu.a -L/usr/local/cuda/lib64 -lcudart_static -pthread -lm  -fopenmp -lrt
cd jvm-packages
export LD_LIBRARY_PATH=$CUDA_HOME:$LD_LIBRARY_PATH
mvn -Dmaven.test.skip=true -DskipTests package
