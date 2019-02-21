======================
XGBoost Python Package
======================

|PyPI version|

Notes
=====

- Windows users: pip installation may not work on some Windows environments, and it may cause unexpected errors.
  
  Installation from pip on Windows is therefore currently disabled for further investigation; please `install from Github <https://xgboost.readthedocs.io/en/latest/build.html>`_ instead.
- If you want to run XGBoost process in parallel using the fork backend for joblib/multiprocessing, you must build XGBoost without support for OpenMP by ``make no_omp=1``. Otherwise, use the forkserver (in Python 3.4) or spawn backend. See the `sklearn\_parallel.py <../demo/guide-python/sklearn_parallel.py>`__ demo.

Requirements
============

Since this package contains C++ source code, ``pip`` needs a C++ compiler from the system to compile the source code on-the-fly.

macOS
-----

On macOS, ``gcc@5`` is required as later versions remove support for OpenMP. `See here <https://github.com/dmlc/xgboost/issues/1501#issuecomment-292209578>`_ for more info.

Please install ``gcc@5`` from `Homebrew <https://brew.sh/>`_::

    brew install gcc@5

After installing ``gcc@5``, set it as your compiler::

    export CC=gcc-5
    export CXX=g++-5

Linux
-----

Please install ``gcc``::

    sudo apt-get install build-essential      # Ubuntu/Debian
    sudo yum groupinstall 'Development Tools' # CentOS/RHEL

Installation
============

From `PyPI <https://pypi.python.org/pypi/xgboost>`_
---------------------------------------------------

For a stable version, install using ``pip``::

    pip install xgboost

From source
-----------

For an up-to-date version, `install from Github <https://xgboost.readthedocs.io/en/latest/build.html>`_:

-  Run ``./build.sh`` in the root of the repo.
-  Make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`_ installed: ``pip install setuptools``
-  Install with ``cd python-package; python setup.py install`` from the root of the repo
-  For Windows users, please use the Visual Studio project file under the `Windows folder <../windows/>`_. See also the `installation
   tutorial <https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13043/run-xgboost-from-windows-and-python>`_ from Kaggle Otto Forum.
-  Add MinGW to the system PATH in Windows if you are using the latest version of xgboost which requires compilation::

    python
    import os
    os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

Examples
========

-  Refer also to the walk through example in `demo folder <https://github.com/dmlc/xgboost/tree/master/demo/guide-python>`_.
-  See also the `example scripts <https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs>`_ for Kaggle
   Higgs Challenge, including `speedtest script <https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs/speedtest.py>`_ on this dataset.

.. |PyPI version| image:: https://badge.fury.io/py/xgboost.svg
   :target: http://badge.fury.io/py/xgboost
