## Build Python package xgboost-cpu (minimal install)

$ErrorActionPreference = "Stop"

. tests/buildkite/conftest.ps1

Write-Host "--- Build libxgboost on Windows (minimal)"

mkdir build
cd build
cmake .. -G"Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
cmake --build . --config Release -- /m /nodeReuse:false `
  "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ..
# Patch to rename pkg to xgboost-cpu
Get-Content tests/buildkite/remove_nccl_dep.patch | patch -p0
Get-Content tests/buildkite/cpu_only_pypkg.patch | patch -p0

cd python-package
conda activate
& pip install --user -v "pip>=23"
& pip --version
& pip wheel --no-deps -v . --wheel-dir dist/
Get-ChildItem . -Filter dist/*.whl |
Foreach-Object {
  & python ../tests/ci_build/rename_whl.py `
    --wheel-path $_.FullName `
    --commit-hash $Env:BUILDKITE_COMMIT `
    --platform-tag win_amd64
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}

Write-Host "--- Upload Python wheel"
cd ..
Get-ChildItem . -Filter python-package/dist/*.whl |
Foreach-Object {
  & buildkite-agent artifact upload python-package/dist/$_
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}
if ( $is_release_branch -eq 1 ) {
  Get-ChildItem . -Filter python-package/dist/*.whl |
  Foreach-Object {
    & aws s3 cp python-package/dist/$_ s3://xgboost-nightly-builds/$Env:BUILDKITE_BRANCH/ `
      --acl public-read --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
  }
}
