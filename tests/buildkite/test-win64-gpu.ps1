$ErrorActionPreference = "Stop"

. tests/buildkite/conftest.ps1

Write-Host "--- Test XGBoost on Windows with CUDA"

New-Item python-package/dist -ItemType Directory -ea 0
New-Item build -ItemType Directory -ea 0
buildkite-agent artifact download "python-package/dist/*.whl" . --step build-win64-gpu
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
buildkite-agent artifact download "build/testxgboost.exe" . --step build-win64-gpu
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
buildkite-agent artifact download "xgboost.exe" . --step build-win64-gpu
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

nvcc --version

Write-Host "--- Run Google Tests"
& build/testxgboost.exe
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Set up Python env"
conda activate
$env_name = -join("win64_", (New-Guid).ToString().replace("-", ""))
mamba env create -n ${env_name} --file=tests/ci_build/conda_env/win64_test.yml
conda activate ${env_name}
Get-ChildItem . -Filter python-package/dist/*.whl |
Foreach-Object {
  & python -m pip install python-package/dist/$_
  if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
}

Write-Host "--- Run Python tests"
python -X faulthandler -m pytest -v -s -rxXs --fulltrace tests/python
Write-Host "--- Run Python tests with GPU"
python -X faulthandler -m pytest -v -s -rxXs --fulltrace -m "(not slow) and (not mgpu)"`
  tests/python-gpu
