## Adopted from https://github.com/chorrell/packer-aws-windows-openssh/blob/20c40aa60b54469b3d85650a2e2e45e35ed83bc7/files/InstallChoco.ps1
## Author: Christopher Horrell (https://github.com/chorrell)

$ErrorActionPreference = "Stop"

# Install Chocolatey
# See https://chocolatey.org/install#individual
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))

# Globally Auto confirm every action
# See: https://docs.chocolatey.org/en-us/faqs#why-do-i-have-to-confirm-packages-now-is-there-a-way-to-remove-this
choco feature enable -n allowGlobalConfirmation
