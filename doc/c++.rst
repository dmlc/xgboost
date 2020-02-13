###############
XGBoost C++ API
###############

Starting from 1.0 release, CMake will generate installation rules to export all C++ headers. But
the c++ interface is much closer to the internal of XGBoost than other language bindings.
As a result it's changing quite often and we don't maintain its stability.  Along with the
plugin system (see ``plugin/example`` in XGBoost's source tree), users can utilize some
existing c++ headers for gaining more access to the internal of XGBoost.

* `C++ interface documentation (latest master branch) <https://xgboost.readthedocs.io/en/latest/dev/files.html>`_
* `C++ interface documentation (last stable release) <https://xgboost.readthedocs.io/en/stable/dev/files.html>`_
