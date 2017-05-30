XGBoost Python Package
======================

|PyPI version|

Installation
------------

We are on `PyPI <https://pypi.python.org/pypi/xgboost>`__ now. For
stable version, please install using pip:

-  ``pip install xgboost``
-  Since this package contains C++ source code, ``pip`` needs a C++ compiler from the system
   to compile the source code on-the-fly. Please follow the following instruction for each
   supported platform.
-  Note for Mac OS X users: please install ``gcc`` from ``brew`` by 
   ``brew tap homebrew/versions; brew install gcc --without-multilib`` firstly.
-  Note for Linux users: please install ``gcc`` by ``sudo apt-get install build-essential`` firstly
   or using the corresponding package manager of the system.
-  Note for windows users: this pip installation may not work on some
   windows environment, and it may cause unexpected errors. pip
   installation on windows is currently disabled for further
   investigation, please install from github.

For up-to-date version, please install from github.

-  To make the python module, type ``./build.sh`` in the root directory
   of project
-  Make sure you have
   `setuptools <https://pypi.python.org/pypi/setuptools>`__
-  Install with ``cd python-package; python setup.py install`` from this directory.
-  For windows users, please use the Visual Studio project file under
   `windows folder <../windows/>`__. See also the `installation
   tutorial <https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13043/run-xgboost-from-windows-and-python>`__
   from Kaggle Otto Forum.
-  Add MinGW to the system PATH in Windows if you are using the latest version of xgboost which requires compilation:

    ```python
    import os
    os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
    ```

Examples
--------

-  Refer also to the walk through example in `demo
   folder <https://github.com/dmlc/xgboost/tree/master/demo/guide-python>`__
-  See also the `example scripts <https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs>`__ for Kaggle
   Higgs Challenge, including `speedtest
   script <https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs/speedtest.py>`__ on this dataset.

Note
----

-  If you want to build xgboost on Mac OS X with multiprocessing support
   where clang in XCode by default doesn't support, please install gcc
   4.9 or higher using `homebrew <http://brew.sh/>`__
   ``brew tap homebrew/versions; brew install gcc --without-multilib``
-  If you want to run XGBoost process in parallel using the fork backend
   for joblib/multiprocessing, you must build XGBoost without support
   for OpenMP by ``make no_omp=1``. Otherwise, use the forkserver (in
   Python 3.4) or spawn backend. See the
   `sklearn\_parallel.py <../demo/guide-python/sklearn_parallel.py>`__
   demo.

.. |PyPI version| image:: https://badge.fury.io/py/xgboost.svg
   :target: http://badge.fury.io/py/xgboost
