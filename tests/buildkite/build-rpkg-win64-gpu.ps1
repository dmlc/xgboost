$ErrorActionPreference = "Stop"

. tests/buildkite/conftest.ps1

Write-Host "--- Build XGBoost R package with CUDA"

nvcc --version
$arch_flag = "-DGPU_COMPUTE_VER=75"

bash tests/ci_build/build_r_pkg_with_cuda_win64.sh $Env:BUILDKITE_COMMIT
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

if ( $is_release_branch -eq 1 ) {
  Write-Host "--- Upload R tarball"
  Get-ChildItem . -Filter xgboost_r_gpu_win64_*.tar.gz |
  Foreach-Object {
    & aws s3 cp $_ s3://xgboost-nightly-builds/$Env:BUILDKITE_BRANCH/ `
    --acl public-read --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
  }
}
