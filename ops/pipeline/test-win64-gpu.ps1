param([string]$CondaEnv)

$ErrorActionPreference = "Stop"

Write-Host "--- Test XGBoost on Windows with CUDA"

nvcc --version

Write-Host "--- Run Google Tests"
build/testxgboost.exe
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Set up Python env"
if (-not $CondaEnv) {
  conda activate
  $CondaEnv = -join("win64_", (New-Guid).ToString().replace("-", ""))
  mamba env create -n $CondaEnv --file=ops/conda_env/win64_test.yml
  if ($LASTEXITCODE -ne 0) { throw "Creating test environment failed" }
}
conda activate $CondaEnv
if ($LASTEXITCODE -ne 0) { throw "Activating test environment failed" }
python -m pip install `
  (Get-ChildItem python-package/dist/*.whl | Select-Object -Expand FullName)
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Run Python tests"
python -X faulthandler -m pytest -v -s -rxXs tests/python
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
Write-Host "--- Run Python tests with GPU"
python -X faulthandler -m pytest -v -s -rxXs -m "(not slow) and (not mgpu)"`
  tests/python-gpu
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
