.. _release:

XGBoost Release Policy
=======================

Versioning Policy
---------------------------

Starting from XGBoost 1.0.0, each XGBoost release will be versioned as [MAJOR].[FEATURE].[MAINTENANCE]

* MAJOR: We gurantee the API compatibility across releases with the same major version number. We expect to have a 1+ years development period for a new MAJOR release version.
* FEATURE: We ship new features, improvements and bug fixes through feature releases. The cycle length of a feature is decided by the size of feature roadmap. The roadmap is decided right after the previous release.
* MAINTENANCE: Maintenance version only contains bug fixes. This type of release only occurs when we found significant correctness and/or performance bugs and barrier for users to upgrade to a new version of XGBoost smoothly.

.. note:: Binary-code compatibility within the same MAJOR version

  In addition to API compatibility guarantee, we also make guarantee with regards to compiled binary code. That is, all programs that uses the external facing functions available from the XGBoost shared library (``libxgboost.so``, ``libxgboost.dylib``, or ``xgboost.dll``) will continue to work with future FEATURE or MAINTENANCE releases of the XGBoost shared library. Refer to `this article <https://en.wikipedia.org/wiki/Binary-code_compatibility>`_ to learn more about binary-code compatibility.

  The binary-code compatibility applies exclusively to the functions listed in the header `include/xgboost/c_api.h <https://xgboost.readthedocs.io/en/latest/dev/c__api_8h.html>`_. As long as your application uses only these functions, it will continue to function with future FEATURE or MAINTENANCE releases of the XGBoost shared library. However, this guarantee will no longer hold if your applications uses functions or constructs not listed in `include/xgboost/c_api.h <https://xgboost.readthedocs.io/en/latest/dev/c__api_8h.html>`_.

  Also note that a change in the MAJOR version signifies a breaking change in the binary. Thus, an application designed to use the shared library ``libxgboost.so`` from XGBoost 1.x is not guaranteed to work with ``libxgboost.so`` from XGBoost 2.x.
