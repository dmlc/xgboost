## Build XGBoost on Windows (CPU or GPU)
##
## Usage:
##   build-win64.ps1 -variant <cpu|gpu>
##
## Parameters:
##   -variant cpu  - Build CPU-only version (creates xgboost-cpu wheel)
##   -variant gpu  - Build with CUDA support (creates default xgboost wheel, includes gtest)
##
## Examples:
##   # Build CPU wheel (xgboost-cpu)
##   ops/pipeline/build-win64.ps1 -variant cpu
##
##   # Build GPU wheel with CUDA
##   ops/pipeline/build-win64.ps1 -variant gpu

param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("cpu", "gpu")]
  [string]$variant
)

$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

# Build common CMake arguments
$cmake_args = @(
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
  "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
)

if ($variant -eq "gpu") {
  Write-Host "--- Build libxgboost on Windows with CUDA"
  nvcc --version
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

  # Add CUDA-specific flags
  $cmake_args += @(
    "-DUSE_CUDA=ON",
    "-DGOOGLE_TEST=ON",
    "-DUSE_DMLC_GTEST=ON"
  )

  # Only build SM75 for non-release branches (faster CI)
  if ($is_release_branch -eq 0) {
    $cmake_args += "-DGPU_COMPUTE_VER=75"
  }
} else {
  Write-Host "--- Build libxgboost on Windows (CPU, minimal)"
}

# Run CMake configure
mkdir build
cd build
& cmake .. @cmake_args
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Build
cmake --build . -v
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ..

# For CPU variant, rename package to xgboost-cpu
if ($variant -eq "cpu") {
  conda activate
  python ops/script/pypi_variants.py --use-suffix=cpu --require-nccl-dep=na
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}

cd python-package
conda activate
& pip wheel --no-deps -v . --wheel-dir dist/
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

python -m wheel tags --python-tag py3 --abi-tag none `
  --platform win_amd64 --remove `
  (Get-ChildItem dist/*.whl | Select-Object -Expand FullName)
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Upload Python wheel"
cd ..
if ($is_release_branch -eq 1) {
  python ops/pipeline/manage-artifacts.py upload `
    --s3-bucket 'xgboost-nightly-builds' `
    --prefix "$Env:BRANCH_NAME/$Env:GITHUB_SHA" --make-public `
    (Get-ChildItem python-package/dist/*.whl | Select-Object -Expand FullName)
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}
