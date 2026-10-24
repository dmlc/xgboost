## Install the CUDA Toolkit components needed to build XGBoost from NVIDIA's
## redistributable archives.
##
## NVIDIA does not ship a CUDA Toolkit installer for Windows ARM64; the toolkit is only
## available as per-component archives listed in
## https://developer.download.nvidia.com/compute/cuda/redist/redistrib_<version>.json.
## This script downloads the components, verifies their SHA-256 checksums and merges
## them into a single toolkit directory.
##
## Usage:
##   ops/pipeline/install-cuda-redist.ps1 -Version 13.4.3 -InstallDir C:\cuda
##
## Inside GitHub Actions, CUDA_PATH is exported and the toolkit's bin directory is added
## to PATH for subsequent steps.

param(
  [Parameter(Mandatory=$true)]
  [string]$Version,
  [Parameter(Mandatory=$true)]
  [string]$InstallDir,
  [string]$Platform = "windows-arm64",
  [string[]]$Components = @("cuda_nvcc", "cuda_cudart", "cuda_crt", "cccl", "libnvvm")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$base_url = "https://developer.download.nvidia.com/compute/cuda/redist"
$InstallDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InstallDir)
$tmp_dir = Join-Path ([IO.Path]::GetTempPath()) "cuda-redist-$Version-$Platform"

Write-Host "--- Install CUDA $Version ($Platform) into $InstallDir"
$manifest = Invoke-RestMethod -Uri "$base_url/redistrib_$Version.json"
New-Item -ItemType Directory -Force -Path $InstallDir, $tmp_dir | Out-Null

foreach ($component in $Components) {
  $entry = $manifest.$component.$Platform
  if (-not $entry) {
    throw "CUDA $Version does not provide $component for $Platform."
  }
  Write-Host "Downloading $component $($manifest.$component.version)"
  $archive = Join-Path $tmp_dir (Split-Path $entry.relative_path -Leaf)
  curl.exe -fsSL --retry 5 --retry-all-errors -o $archive "$base_url/$($entry.relative_path)"
  if ($LASTEXITCODE -ne 0) { throw "Failed to download $component" }

  $sha256 = (Get-FileHash -Path $archive -Algorithm SHA256).Hash
  if ($sha256 -ne $entry.sha256) {
    throw "SHA-256 mismatch for ${component}: expected $($entry.sha256), got $sha256"
  }

  $extract_dir = Join-Path $tmp_dir $component
  Expand-Archive -Path $archive -DestinationPath $extract_dir -Force
  # Each archive holds a single top-level directory with the usual toolkit layout
  # (bin, include, lib, ...), so merging them yields a regular toolkit tree.
  $archive_root = Get-ChildItem -Path $extract_dir -Directory | Select-Object -First 1
  Copy-Item -Path (Join-Path $archive_root.FullName "*") -Destination $InstallDir -Recurse -Force
}
Remove-Item -Recurse -Force $tmp_dir

& (Join-Path $InstallDir "bin\nvcc.exe") --version
if ($LASTEXITCODE -ne 0) { throw "nvcc is not functional" }

if ($Env:GITHUB_ACTIONS) {
  Add-Content -Path $Env:GITHUB_ENV -Value "CUDA_PATH=$InstallDir"
  Add-Content -Path $Env:GITHUB_PATH -Value (Join-Path $InstallDir "bin")
}
