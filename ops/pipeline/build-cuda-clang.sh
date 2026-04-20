#!/bin/bash
## Build XGBoost with clang as the CUDA compiler.
## Intended as compile-only CI coverage for the clang-CUDA path.

set -euo pipefail

clang_version="21.1.8"
cmake_version="4.2.3"
build_dir="build-clang-cuda"
target="xgboost"
configure_only=0
jobs="${XGBOOST_BUILD_JOBS:-}"
cmake_prefix_path="${XGBOOST_CMAKE_PREFIX_PATH:-/opt/grpc}"
gpu_compute_ver="${XGBOOST_GPU_COMPUTE_VER:-75}"
cuda_toolkit_root="${XGBOOST_CUDA_TOOLKIT_ROOT:-/usr/local/cuda}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clang-version)
      clang_version="$2"
      shift 2
      ;;
    --cmake-version)
      cmake_version="$2"
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
      echo "Usage: $0 [--clang-version <version>] [--cmake-version <version>] [--build-dir <dir>] [--target <target>] [--configure-only] [--jobs <n>]"
      exit 1
      ;;
  esac
done

if [[ "${build_dir}" != /* ]]; then
  build_dir="${repo_root}/${build_dir}"
fi

if [[ "${XGBOOST_SKIP_CLANG_INSTALL:-0}" != 1 ]]; then
  if command -v mamba >/dev/null 2>&1; then
    mamba install -y -n base -c conda-forge "clangxx=${clang_version}" "clang-tools=${clang_version}" "cmake=${cmake_version}"
  elif command -v conda >/dev/null 2>&1; then
    conda install -y -n base -c conda-forge "clangxx=${clang_version}" "clang-tools=${clang_version}" "cmake=${cmake_version}"
  else
    echo "clangxx=${clang_version}, clang-tools=${clang_version}, and cmake=${cmake_version} are required, but neither mamba nor conda is available."
    exit 1
  fi
fi

if [[ -n "${XGBOOST_CLANG_PREFIX:-}" ]]; then
  clang_prefix="${XGBOOST_CLANG_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  clang_prefix="$(conda info --base)"
else
  if ! clang_path="$(command -v clang++)"; then
    echo "Could not find clang++ on PATH."
    echo "Install the clang toolchain, set XGBOOST_CLANG_PREFIX, or ensure clang++ is available on PATH."
    exit 1
  fi
  clang_prefix="$(cd "$(dirname "${clang_path}")/.." && pwd)"
fi

clang_bin_dir="${clang_prefix}/bin"
clang_c="${clang_bin_dir}/clang"
clang_cxx="${clang_bin_dir}/clang++"
cmake_bin="${clang_bin_dir}/cmake"

if [[ ! -x "${clang_c}" || ! -x "${clang_cxx}" || ! -x "${cmake_bin}" ]]; then
  echo "Could not find clang/CMake toolchain under ${clang_prefix}"
  exit 1
fi

export PATH="${clang_bin_dir}:${PATH}"

echo "--- Build with clang-CUDA using ${clang_cxx}"
"${clang_c}" --version
"${clang_cxx}" --version
"${cmake_bin}" --version

if ! command -v clang-linker-wrapper >/dev/null 2>&1; then
  echo "clang-linker-wrapper is required for clang CUDA offload linking. Install clang-tools=${clang_version}."
  exit 1
fi
if [[ -f "${clang_bin_dir}/x86_64-conda-linux-gnu-clang++.cfg" ]] &&
  ! command -v x86_64-conda-linux-gnu-ld >/dev/null 2>&1; then
  echo "x86_64-conda-linux-gnu-ld is required for the conda clang toolchain."
  exit 1
fi

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

cmake_args=(
  -GNinja
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DCMAKE_PREFIX_PATH="${cmake_prefix_path}"
  -DCMAKE_C_COMPILER="${clang_c}"
  -DCMAKE_CXX_COMPILER="${clang_cxx}"
  -DCMAKE_CUDA_COMPILER="${clang_cxx}"
  -DCMAKE_CUDA_ARCHITECTURES="${gpu_compute_ver}"
  -DCUDAToolkit_ROOT="${cuda_toolkit_root}"
  -DUSE_CUDA=ON
  -DUSE_OPENMP=ON
  -DHIDE_CXX_SYMBOLS=ON
  -DUSE_NCCL=OFF
  -DENABLE_ALL_WARNINGS=ON
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
  -DPLUGIN_FEDERATED=OFF
  -DGPU_COMPUTE_VER="${gpu_compute_ver}"
)
cmake_args+=("${launcher_args[@]}")
"${cmake_bin}" -S "${repo_root}" -B "${build_dir}" "${cmake_args[@]}"

if [[ "${configure_only}" == 1 ]]; then
  exit 0
fi

time "${cmake_bin}" --build "${build_dir}" --target "${target}" --verbose -j "${jobs}"
