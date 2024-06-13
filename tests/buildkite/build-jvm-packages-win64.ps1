$ErrorActionPreference = "Stop"

. tests/buildkite/conftest.ps1

Write-Host "--- Build and test JVM packages on Windows"

mkdir build
cd build
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-11.0.23.9-hotspot\"
cmake .. -G"Visual Studio 17 2022" -A x64 -DJVM_BINDINGS=ON
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
cmake --build . --config Release -- /m /nodeReuse:false `
  "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Write-Host "--- Stash xgboost4j.dll"
cd ../lib
buildkite-agent artifact upload "xgboost4j.dll"
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

Copy-Item -Path xgboost4j.dll -Destination xgboost4j_$Env:BUILDKITE_COMMIT.dll
aws s3 cp xgboost4j_$Env:BUILDKITE_COMMIT.dll `
  s3://xgboost-nightly-builds/$Env:BUILDKITE_BRANCH/ `
  --acl public-read --no-progress
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }

# TODO(hcho3): Move this to the bootstrap script
choco install maven

Write-Host "--- Test XGBoost4J (Core)"
cd ..
New-Item -ItemType Directory -Path jvm-packages/xgboost4j/src/main/resources/lib/windows/x86_64 -Force
Copy-Item -Path lib/xgboost4j.dll -Destination jvm-packages/xgboost4j/src/main/resources/lib/windows/x86_64/xgboost4j.dll
cd jvm-packages
& C:\ProgramData\chocolatey\lib\maven\apache-maven-3.9.7\mvn test -B -pl :xgboost4j_2.12
if ($LASTEXITCODE -ne 0) { throw "Last command failed" }
