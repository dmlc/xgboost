[CmdletBinding()]
Param(
    [Parameter(
      Mandatory=$true,
      Position=0
    )][string]$command,
    [Parameter(
      Mandatory=$true,
      Position=1
    )][string]$remote_prefix,
    [Parameter(
      Mandatory=$true,
      Position=2,
      ValueFromRemainingArguments=$true
    )][string[]]$artifacts
)

## Convenience wrapper for ops/pipeline/stash-artifacts.py
## Meant to be used inside GitHub Actions

$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

foreach ($env in "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "RUNS_ON_S3_BUCKET_CACHE") {
  $val = [Environment]::GetEnvironmentVariable($env)
  if ($val -eq $null) {
    Write-Host "Error: $env must be set."
    exit 1
  }
}

$artifact_stash_prefix = "cache/${Env:GITHUB_REPOSITORY}/stash/${Env:GITHUB_RUN_ID}"

conda activate

Write-Host @"
python ops/pipeline/stash-artifacts.py `
  --command "${command}"  `
  --s3-bucket "${Env:RUNS_ON_S3_BUCKET_CACHE}" `
  --prefix "${artifact_stash_prefix}/${remote_prefix}" `
  -- $artifacts
"@
python ops/pipeline/stash-artifacts.py `
  --command "${command}"  `
  --s3-bucket "${Env:RUNS_ON_S3_BUCKET_CACHE}" `
  --prefix "${artifact_stash_prefix}/${remote_prefix}" `
  -- $artifacts
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
