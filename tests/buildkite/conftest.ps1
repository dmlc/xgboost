if ( $Env:BUILDKITE_PULL_REQUEST -and ($Env:BUILDKITE_PULL_REQUEST -ne "false") ) {
  $is_pull_request = 1
} else {
  $is_pull_request = 0
}

if ( ($Env:BUILDKITE_BRANCH -eq "master") -or ($Env:BUILDKITE_BRANCH -match "release_.+") ) {
  $is_release_branch = 1
  $enforce_daily_budget = 0
} else {
  $is_release_branch = 0
  $enforce_daily_budget = 1
}
