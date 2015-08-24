XGBoost Python Package
======================
Installation
------------
We are on [PyPI](https://pypi.python.org/pypi/xgboost) now. For stable version, please install using pip:

* ```pip install xgboost```
* Note for windows users: this pip installation may not work on some windows environment, and it may cause unexpected errors. pip installation on windows is currently disabled for further invesigation, please install from github.

For up-to-date version, please install from github.

* To make the python module, type ```./build.sh``` in the root directory of project
* Make sure you have [setuptools](https://pypi.python.org/pypi/setuptools)
* Install with `python setup.py install` from this directory.
* For windows users, please use the Visual Studio project file under [windows folder](../windows/). See also the [installation tutorial](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13043/run-xgboost-from-windows-and-python) from Kaggle Otto Forum.

Examples
------

* Refer also to the walk through example in [demo folder](../demo/guide-python)
* See also the [example scripts](../demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](../demo/kaggle-higgs/speedtest.py) on this dataset.

Note
-----

* If you want to build xgboost on Mac OS X with multiprocessing support where clang in XCode by default doesn't support, please install gcc 4.9 or higher using [homebrew](http://brew.sh/) ```brew tap homebrew/versions; brew install gcc49```
* If you want to run XGBoost process in parallel using the fork backend for joblib/multiprocessing, you must build XGBoost without support for OpenMP by `make no_omp=1`. Otherwise, use the forkserver (in Python 3.4) or spawn backend. See the [sklearn_parallel.py](../demo/guide-python/sklearn_parallel.py) demo.
