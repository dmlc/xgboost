The solution has been created with Visual Studio Express 2010.

How to Build Windows Version
=====
* Open the solution file with Visual Studio
* Select x64 and Release in build
* Rebuild all

This should give you xgboost.exe for CLI version and xgboost_wrapper.dll for python

Use Python Module
=====
* After you build the dll, you can simply add the path to [../wrapper](../wrapper) to sys.path and import xgboost
```
sys.path.append('path/to/xgboost/wrapper')
import xgboost as xgb
```
* Alternatively, you can add that path to system enviroment variable ```PYTHONPATH```
  - Doing so allows you to import xgboost directly like other python packages

R Package
====
* see [R-package](../R-package) instead
