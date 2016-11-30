echo "copy native library"
set libsource=..\lib\libxgboost4j.so

if not exist %libsource% (
goto end
)

set libfolder=src\main\resources\lib
set libpath=%libfolder%\xgboost4j.dll
if not exist %libfolder% (mkdir %libfolder%)
if exist %libpath% (del %libpath%)
copy %libsource% %libpath%
echo complete
exit

:end
  echo "source library not found, please build it first by runing mingw32-make jvm"
  pause
  exit
