$ErrorActionPreference = "Stop"

Write-Host "--- Test XGBoost on Windows with CUDA"

nvcc --version

Write-Host "--- Run Google Tests"
build/testxgboost.exe
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Set up Python env"
conda activate
$env_name = -join("win64_", (New-Guid).ToString().replace("-", ""))
mamba env create -n ${env_name} --file=ops/conda_env/win64_test.yml
conda activate ${env_name}
python -m pip install `
  (Get-ChildItem python-package/dist/*.whl | Select-Object -Expand FullName)
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Run Python tests"
python -X faulthandler -m pytest -v -s -rxXs --fulltrace tests/python
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
Write-Host "--- Run Python tests with GPU"
python -X faulthandler -m pytest -v -s -rxXs --fulltrace -m "(not slow) and (not mgpu)"`
  tests/python-gpu
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
