## Adopted from https://github.com/chorrell/packer-aws-windows-openssh/blob/20c40aa60b54469b3d85650a2e2e45e35ed83bc7/files/PrepareImage.ps1
## Author: Christopher Horrell (https://github.com/chorrell)

$ErrorActionPreference = "Stop"

Write-Output "Cleaning up keys"
$openSSHAuthorizedKeys = Join-Path $env:ProgramData "ssh\administrators_authorized_keys"
Remove-Item -Recurse -Force -Path $openSSHAuthorizedKeys

# Make sure task is enabled
Enable-ScheduledTask "DownloadKey"

Write-Output "Running Sysprep"
& "$Env:Programfiles\Amazon\EC2Launch\ec2launch.exe" sysprep
