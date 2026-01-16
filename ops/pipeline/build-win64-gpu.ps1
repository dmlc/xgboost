## Build Python package and gtest with GPU enabled

$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

Write-Host "--- Build libxgboost on Windows with CUDA"

nvcc --version
if ( $is_release_branch -eq 0 ) {
  $arch_flag = "-DGPU_COMPUTE_VER=75"
} else {
  $arch_flag = ""
}

mkdir build
cd build
cmake .. -G"Ninja" -DUSE_CUDA=ON `
  -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache `
  -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON `
  ${arch_flag}
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

cmake --build . --config Release -- /m /nodeReuse:false `
  "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ../python-package
conda activate
pip wheel --no-deps -v . --wheel-dir dist/
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
python -m wheel tags --python-tag py3 --abi-tag none `
  --platform win_amd64 --remove `
  (Get-ChildItem dist/*.whl | Select-Object -Expand FullName)
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Upload Python wheel"
cd ..
if ( $is_release_branch -eq 1 ) {
  python ops/pipeline/manage-artifacts.py upload `
    --s3-bucket 'xgboost-nightly-builds' `
    --prefix "$Env:BRANCH_NAME/$Env:GITHUB_SHA" --make-public `
    (Get-ChildItem python-package/dist/*.whl | Select-Object -Expand FullName)
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}
