XGBoost Plugins Modules
=======================
This folder contains plugin modules to xgboost that can be optionally installed.
The plugin system helps us to extend xgboost with additional features,
and add experimental features that may not yet ready to be included in main project.

To include a certain plugin, say ```plugin_a```, you only need to add the following line to the config.mk.

```makefile
# Add plugin by include the plugin in config
XGB_PLUGINS += plugin/plugin_a/plugin.mk
```

Then rebuild libxgboost by typing make, you can get a new library with the plugin enabled.

Link Static XGBoost Library with Plugins
----------------------------------------
This problem only happens when you link ```libxgboost.a```.
If you only use ```libxgboost.so```(this include python and other bindings),
you can ignore this section.

When you want to link ```libxgboost.a``` with additional plugins included,
you will need to enabled whole archeive via The following option.
```bash
--whole-archive libxgboost.a --no-whole-archive
```

Write Your Own Plugin
---------------------
You can plugin your own modules to xgboost by adding code to this folder,
without modification to the main code repo.
The [example](example) folder provides an example to write a plugin.
