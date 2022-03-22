XGBoost Plugins Modules
=======================

This folder contains plugin modules to xgboost that can be optionally installed.  The
plugin system helps us to extend xgboost with additional features, and add experimental
features that may not yet be ready to be included in the main project.

To include a certain plugin, say ```plugin_a```, you only need to add the following line
to `xgboost/plugin/CMakeLists.txt`
``` cmake
set(PLUGIN_SOURCES ${PLUGIN_SOURCES}
    ${xgboost_SOURCE_DIR}/plugin/plugin_a.cc PARENT_SCOPE)
```
along with specified source file `plugin_a.cc`.

Then rebuild XGBoost with CMake.

Write Your Own Plugin
---------------------
You can plugin your own modules to xgboost by adding code to this folder,
without modification to the main code repo.
The [example](example) folder provides an example to write a plugin.

List of register functions
--------------------------
A plugin has to register a new functionality to xgboost to be able to use it.
The register macros available to plugin writers are:

 - XGBOOST_REGISTER_METRIC - Register an evaluation metric
 - XGBOOST_REGISTER_GBM - Register a new gradient booster that learns through
   gradient statistics
 - XGBOOST_REGISTER_OBJECTIVE - Register a new objective function used by xgboost
 - XGBOOST_REGISTER_TREE_UPDATER - Register a new tree-updater which updates
   the tree given the gradient information

And from dmlc-core:

 - DMLC_REGISTER_PARAMETER - Register a set of parameter for a specific usecase
 - DMLC_REGISTER_DATA_PARSER - Register a data parser where the data can be
   represented by a URL. This is used by DMatrix.
