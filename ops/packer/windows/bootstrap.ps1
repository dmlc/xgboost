## Install packages from Chocolatey

# jq & yq
Write-Output "Installing jq and yq..."
choco install jq --version=1.7.1
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
choco install yq --version=4.40.2
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# AWS CLI
Write-Output "Installing AWS CLI..."
choco install awscli --version=2.18.11
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Git
Write-Host '>>> Installing Git...'
choco install git --version=2.47.0
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# CMake
Write-Host '>>> Installing CMake 3.30.5...'
choco install cmake --version 3.30.5 --installargs "ADD_CMAKE_TO_PATH=System"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Notepad++
Write-Host '>>> Installing Notepad++...'
choco install notepadplusplus
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Miniforge3
Write-Host '>>> Installing Miniforge3...'
choco install miniforge3 --params="'/InstallationType:AllUsers /RegisterPython:1 /D:C:\tools\miniforge3'"
C:\tools\miniforge3\Scripts\conda.exe init --user --system
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
. "C:\Windows\System32\WindowsPowerShell\v1.0\profile.ps1"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
conda config --set auto_activate_base false

# Java 11
Write-Host '>>> Installing Java 11...'
choco install openjdk11
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Maven
Write-Host '>>> Installing Maven...'
choco install maven
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# GraphViz
Write-Host '>>> Installing GraphViz...'
choco install graphviz
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# Visual Studio 2022 Community
Write-Host '>>> Installing Visual Studio 2022 Community...'
choco install visualstudio2022community `
    --params "--wait --passive --norestart"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
choco install visualstudio2022-workload-nativedesktop --params `
    "--wait --passive --norestart --includeOptional"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# CUDA 12.5
Write-Host '>>> Installing CUDA 12.5...'
choco install cuda --version=12.5.1.555
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# R 4.3
Write-Host '>>> Installing R...'
choco install r.project --version=4.3.2
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
choco install rtools --version=4.3.5550
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
