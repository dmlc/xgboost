## Build Python package xgboost-cpu (minimal install)

$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

choco install gzip

Write-Host "--- Build libxgboost on Windows (minimal)"

mkdir build
cd build
cmake .. -G"Ninja" -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

cmake --build . -v
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ..
# Patch to rename pkg to xgboost-cpu
conda activate
python ops/script/pypi_variants.py --use-suffix=cpu --require-nccl-dep=na
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

cd python-package
& pip wheel --no-deps -v . --wheel-dir dist/
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
