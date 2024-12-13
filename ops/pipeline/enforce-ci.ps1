## Ensure that a script is running inside the CI.
## Usage: . ops/pipeline/enforce-ci.ps1

if ( -Not $Env:GITHUB_ACTIONS ) {
  $script_name = (Split-Path -Path $PSCommandPath -Leaf)
  Write-Host "$script_name is not meant to run locally; it should run inside GitHub Actions."
  Write-Host "Please inspect the content of $script_name and locate the desired command manually."
  exit 1
}

if ( -Not $Env:BRANCH_NAME ) {
  Write-Host "Make sure to define environment variable BRANCH_NAME."
  exit 2
}

if ( $Env:GITHUB_BASE_REF ) {
  $is_pull_request = 1
} else {
  $is_pull_request = 0
}

if ( ($Env:BRANCH_NAME -eq "master") -or ($Env:BRANCH_NAME -match "release_.+") ) {
  $is_release_branch = 1
  $enforce_daily_budget = 0
} else {
  $is_release_branch = 0
  $enforce_daily_budget = 1
}
