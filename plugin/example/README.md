XGBoost Plugin Example
======================
This folder provides an example of implementing xgboost plugin.

There are three steps you need to do to add a plugin to xgboost
- Create your source .cc file, implement a new extension
  - In this example [custom_obj.cc](custom_obj.cc)
- Register this extension to xgboost via a registration macro
  - In this example ```XGBOOST_REGISTER_OBJECTIVE``` in [this line](custom_obj.cc#L78)
- Add a line to `xgboost/plugin/CMakeLists.txt`:
```
target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc)
```

Then you can test this plugin by using ```objective=mylogistic``` parameter.

<!--  LocalWords:  XGBoost
 -->
