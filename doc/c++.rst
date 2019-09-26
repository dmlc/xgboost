###############
XGBoost C++ API
###############

We installs all c++ headers during ``CMake`` build installation after 1.0 release.  But
the c++ interface is much closer to the internal of XGBoost than other language bindings.
As a result it's changing quite often and we don't maintain its stability.  Along with the
plugin system (see ``plugin/example`` in XGBoost's source tree), users can utilize some
existing c++ headers for gaining more access to the internal of XGBoost.
