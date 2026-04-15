#!/bin/bash
## Build XGBoost with clang as the CUDA compiler.
## Intended as compile-only CI coverage for the clang-CUDA path.

set -euo pipefail

clang_version="21.1.8"
build_dir="build-clang-cuda"
target="xgboost"
configure_only=0
jobs="${XGBOOST_BUILD_JOBS:-}"
cmake_prefix_path="${XGBOOST_CMAKE_PREFIX_PATH:-/opt/grpc}"
nccl_include_dir="${XGBOOST_NCCL_INCLUDE_DIR:-/usr/include}"
nccl_root="${XGBOOST_NCCL_ROOT:-}"
gpu_compute_ver="${XGBOOST_GPU_COMPUTE_VER:-75}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clang-version)
      clang_version="$2"
      shift 2
      ;;
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --target)
      target="$2"
      shift 2
      ;;
    --configure-only)
      configure_only=1
      shift
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 [--clang-version <version>] [--build-dir <dir>] [--target <target>] [--configure-only] [--jobs <n>]"
      exit 1
      ;;
  esac
done

if [[ "${XGBOOST_SKIP_CLANG_INSTALL:-0}" != 1 ]]; then
  if command -v mamba >/dev/null 2>&1; then
    mamba install -y -n base -c conda-forge "clangxx=${clang_version}"
  elif command -v conda >/dev/null 2>&1; then
    conda install -y -n base -c conda-forge "clangxx=${clang_version}"
  else
    echo "clangxx=${clang_version} is required, but neither mamba nor conda is available."
    exit 1
  fi
fi

if [[ -n "${XGBOOST_CLANG_PREFIX:-}" ]]; then
  clang_prefix="${XGBOOST_CLANG_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  clang_prefix="$(conda info --base)"
else
  clang_path="$(command -v clang++)"
  clang_prefix="$(cd "$(dirname "${clang_path}")/.." && pwd)"
fi

clang_bin_dir="${clang_prefix}/bin"
clang_c="${clang_bin_dir}/clang"
clang_cxx="${clang_bin_dir}/clang++"

if [[ ! -x "${clang_c}" || ! -x "${clang_cxx}" ]]; then
  echo "Could not find clang toolchain under ${clang_prefix}"
  exit 1
fi

echo "--- Build with clang-CUDA using ${clang_cxx}"
"${clang_c}" --version
"${clang_cxx}" --version

if [[ -z "${jobs}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
  else
    jobs=4
  fi
  if (( jobs > 35 )); then
    jobs=35
  fi
fi

launcher_args=()
if command -v sccache >/dev/null 2>&1; then
  launcher_args=(
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache
  )
fi

nccl_args=()
if [[ -n "${nccl_root}" ]]; then
  nccl_args=(
    -DNccl_ROOT="${nccl_root}"
    -DBUILD_WITH_SHARED_NCCL=ON
  )
fi

mkdir -p "${build_dir}"
pushd "${build_dir}"

cmake_args=(
  ..
  -GNinja
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DCMAKE_PREFIX_PATH="${cmake_prefix_path}"
  -DCMAKE_C_COMPILER="${clang_c}"
  -DCMAKE_CXX_COMPILER="${clang_cxx}"
  -DCMAKE_CUDA_COMPILER="${clang_cxx}"
  -DUSE_CUDA=ON
  -DUSE_OPENMP=ON
  -DHIDE_CXX_SYMBOLS=ON
  -DUSE_NCCL=ON
  -DNCCL_INCLUDE_DIR="${nccl_include_dir}"
  -DUSE_DLOPEN_NCCL=ON
  -DGOOGLE_TEST=ON
  -DUSE_DMLC_GTEST=ON
  -DENABLE_ALL_WARNINGS=ON
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
  -DPLUGIN_FEDERATED=OFF
  -DGPU_COMPUTE_VER="${gpu_compute_ver}"
)
cmake_args+=("${nccl_args[@]}")
cmake_args+=("${launcher_args[@]}")
cmake "${cmake_args[@]}"

if [[ "${configure_only}" == 1 ]]; then
  popd
  exit 0
fi

time ninja -v -j "${jobs}" "${target}"

popd
