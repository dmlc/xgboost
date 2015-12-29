echo "move native library"
set libsource=..\windows\x64\Release\xgboost4j.dll

if not exist %libsource% (
goto end
)

set libfolder=xgboost4j\src\main\resources\lib
set libpath=%libfolder%\xgboost4j.dll
if not exist %libfolder% (mkdir %libfolder%)
if exist %libpath% (del %libpath%)
move %libsource% %libfolder%
echo complete
pause
exit

:end
  echo "source library not found, please build it first from ..\windows\xgboost.sln"
  pause
  exit