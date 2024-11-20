$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

Write-Host "--- Build libxgboost on Windows with CUDA"

nvcc --version
if ( $is_release_branch -eq 0 ) {
  $arch_flag = "-DGPU_COMPUTE_VER=75"
} else {
  $arch_flag = ""
}

# Work around https://github.com/NVIDIA/cccl/issues/1956
# TODO(hcho3): Remove this once new CUDA version ships with CCCL 2.6.0+
git clone https://github.com/NVIDIA/cccl.git -b v2.6.1 --quiet
mkdir build
cd build
cmake .. -G"Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON `
  -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DBUILD_DEPRECATED_CLI=ON `
  -DCMAKE_PREFIX_PATH="$(Get-Location)/../cccl" ${arch_flag}
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
cmake --build . --config Release -- /m /nodeReuse:false `
  "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ../python-package
conda activate
pip install --user -v "pip>=23"
pip --version
pip wheel --no-deps -v . --wheel-dir dist/
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
python ../ops/script/rename_whl.py `
    --wheel-path (Get-ChildItem dist/*.whl | Select-Object -Expand FullName) `
    --commit-hash $Env:GITHUB_SHA `
    --platform-tag win_amd64
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Upload Python wheel"
cd ..
if ( $is_release_branch -eq 1 ) {
  aws s3 cp (Get-ChildItem python-package/dist/*.whl | Select-Object -Expand FullName) `
    s3://xgboost-nightly-builds/$Env:BRANCH_NAME/ --acl public-read --no-progress
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}
