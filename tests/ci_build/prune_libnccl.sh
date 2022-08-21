#!/usr/bin/env bash
set -e

rm -rf tmp_nccl

mkdir tmp_nccl
pushd tmp_nccl

set -x

cat << EOF > test.cu
int main(void) { return 0; }
EOF

cat << EOF > CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gencode_extractor CXX C)
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_HOST_COMPILER \${CMAKE_CXX_COMPILER})
enable_language(CUDA)
include(../cmake/Utils.cmake)
set(GEN_CODE "")
format_gencode_flags("" GEN_CODE)
add_library(test OBJECT test.cu)
set_property(TARGET test PROPERTY CUDA_ARCHITECTURES \${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
EOF

cmake . -GNinja
gen_code=$(grep -o -- '--generate-code=\S*' compile_commands.json | paste -sd ' ')

nvprune ${gen_code} /usr/lib64/libnccl_static.a -o ../libnccl_static.a

popd
rm -rf tmp_nccl

set +x
