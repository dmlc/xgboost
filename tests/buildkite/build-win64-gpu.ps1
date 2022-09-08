$ErrorActionPreference = "Stop"

. tests/buildkite/conftest.ps1

Write-Host "--- Build libxgboost on Windows with CUDA"

nvcc --version
if ( $is_release_branch -eq 0 ) {
  $arch_flag = "-DGPU_COMPUTE_VER=75"
} else {
  $arch_flag = ""
}
mkdir build
cd build
cmake .. -G"Visual Studio 15 2017 Win64" -DUSE_CUDA=ON -DCMAKE_VERBOSE_MAKEFILE=ON `
  -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DCMAKE_UNITY_BUILD=ON ${arch_flag}
$msbuild = -join @(
  "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0"
  "\\Bin\\MSBuild.exe"
)
& $msbuild xgboost.sln /m /p:Configuration=Release /nodeReuse:false
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Build binary wheel"
cd ../python-package
conda activate
& python setup.py bdist_wheel --universal
Get-ChildItem . -Filter dist/*.whl |
Foreach-Object {
  & python ../tests/ci_build/rename_whl.py $_.FullName $Env:BUILDKITE_COMMIT win_amd64
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}

Write-Host "--- Insert vcomp140.dll (OpenMP runtime) into the wheel"
cd dist
Copy-Item -Path ../../tests/ci_build/insert_vcomp140.py -Destination .
& python insert_vcomp140.py *.whl
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Upload Python wheel"
cd ../..
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

Write-Host "--- Stash C++ test executables"
& buildkite-agent artifact upload build/testxgboost.exe
& buildkite-agent artifact upload xgboost.exe
