The solution has been created with Visual Studio Express 2010.

How to Build Windows Version
=====
* Open the solution file with Visual Studio
* Select x64 and Release in build
	- For 32bit windows or python, try win32 and Release (not fully tested)
* Rebuild all

This should give you xgboost.exe for CLI version and xgboost_wrapper.dll for python

Use Python Module
=====
* After you build the dll, you can install the Python package from the [../python-package](../python-package) folder

```
python setup.py install
```

And import it as usual

```
import xgboost as xgb
```

R Package
====
* see [R-package](../R-package) instead
