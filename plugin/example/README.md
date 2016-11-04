XGBoost Plugin Example
======================
This folder provides an example of xgboost plugin.

There are three steps you need to do to add a plugin to xgboost
- Create your source .cc file, implement a new extension
  - In this example [custom_obj.cc](custom_obj.cc)
- Register this extension to xgboost via a registration macro
  - In this example ```XGBOOST_REGISTER_OBJECTIVE``` in [this line](custom_obj.cc#L75)
- Create a [plugin.mk](plugin.mk) on this folder

To add this plugin, add the following line to ```config.mk```(template in make/config.mk).
```makefile
# Add plugin by include the plugin in config
XGB_PLUGINS += plugin/plugin_a/plugin.mk
```

Then you can test this plugin by using ```objective=mylogistic``` parameter.



