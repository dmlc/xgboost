[CmdletBinding()]
Param(
    [Parameter(
      Mandatory=$true,
      Position=0,
      ValueFromRemainingArguments=$true
    )][string[]]$artifacts
)

## Convenience wrapper for ops/pipeline/stash-artifacts.py
## Meant to be used inside GitHub Actions

$ENV_VAR_DOC = @'
Inputs
  - COMMAND: Either "upload" or "download"
  - KEY:     Unique string to identify a group of artifacts
'@

$ErrorActionPreference = "Stop"

. ops/pipeline/enforce-ci.ps1

foreach ($env in "COMMAND", "KEY", "GITHUB_REPOSITORY", "GITHUB_RUN_ID",
                 "RUNS_ON_S3_BUCKET_CACHE") {
  $val = [Environment]::GetEnvironmentVariable($env)
  if ($val -eq $null) {
    Write-Host "Error: $env must be set.`n${ENV_VAR_DOC}"
    exit 1
  }
}

$artifact_stash_prefix = "cache/${Env:GITHUB_REPOSITORY}/stash/${Env:GITHUB_RUN_ID}"

conda activate

Write-Host @"
python ops/pipeline/stash-artifacts.py `
  --command "${Env:COMMAND}"  `
  --s3-bucket "${Env:RUNS_ON_S3_BUCKET_CACHE}" `
  --prefix "${artifact_stash_prefix}/${Env:KEY}" `
  -- $artifacts
"@
python ops/pipeline/stash-artifacts.py `
  --command "${Env:COMMAND}"  `
  --s3-bucket "${Env:RUNS_ON_S3_BUCKET_CACHE}" `
  --prefix "${artifact_stash_prefix}/${Env:KEY}" `
  -- $artifacts
