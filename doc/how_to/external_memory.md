Using XGBoost External Memory Version(beta)
===========================================
There is no big difference between using external memory version and in-memory version.
The only difference is the filename format.

The external memory version takes in the following filename format
```
filename#cacheprefix
```

The ```filename``` is the normal path to libsvm file you want to load in, ```cacheprefix``` is a
path to a cache file that xgboost will use for external memory cache.

The following code was extracted from [../demo/guide-python/external_memory.py](../demo/guide-python/external_memory.py)
```python
dtrain = xgb.DMatrix('../data/agaricus.txt.train#dtrain.cache')
```
You can find that there is additional ```#dtrain.cache``` following the libsvm file, this is the name of cache file.
For CLI version, simply use ```"../data/agaricus.txt.train#dtrain.cache"``` in filename.

Performance Note
----------------
* the parameter ```nthread``` should be set to number of ***real*** cores
  - Most modern CPU offer hyperthreading, which means you can have a 4 core cpu with 8 threads
  - Set nthread to be 4 for maximum performance in such case

Distributed Version
-------------------
The external memory mode naturally works on distributed version, you can simply set path like
```
data = "hdfs:///path-to-data/#dtrain.cache"
```
xgboost will cache the data to the local position. When you run on YARN, the current folder is temporal
so that you can directly use ```dtrain.cache``` to cache to current folder.


Usage Note
----------
* This is a experimental version
  - If you like to try and test it, report results to https://github.com/dmlc/xgboost/issues/244
* Currently only importing from libsvm format is supported
  - Contribution of ingestion from other common external memory data source is welcomed
