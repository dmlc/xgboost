#################
XGBoost C Package
#################

XGBoost implements a set of C API designed for various bindings, we maintain its stability
and the CMake/make build interface.  See :doc:`/tutorials/c_api_tutorial` for an
introduction and ``demo/c-api/`` for related examples.  Also one can generate doxygen
document by providing ``-DBUILD_C_DOC=ON`` as parameter to ``CMake`` during build, or
simply look at function comments in ``include/xgboost/c_api.h``. The reference is exported
to sphinx with the help of breathe, which doesn't contain links to examples but might be
easier to read. For the original doxygen pages please visit:

* `C API documentation (latest master branch) <./dev/c__api_8h.html>`_
* `C API documentation (last stable release) <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html>`_

***************
C API Reference
***************

.. contents::
  :backlinks: none
  :local:

Library
=======

.. doxygengroup:: Library
   :project: xgboost

DMatrix
=======

.. doxygengroup:: DMatrix
   :project: xgboost

Streaming
---------

.. doxygengroup:: Streaming
   :project: xgboost

Booster
=======

.. doxygengroup:: Booster
   :project: xgboost

Prediction
----------

.. doxygengroup:: Prediction
   :project: xgboost

Serialization
-------------

.. doxygengroup:: Serialization
   :project: xgboost

Collective
==========

.. doxygengroup:: Collective
   :project: xgboost
