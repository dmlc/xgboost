Data Callback
=============

A simple demo for using custom data iterator with XGBoost.  The primary function for this
is external-memory training with user provided data loaders.  In the example, we have
defined a custom data iterator with 2 methods: `reset` and `next`.  The `next` method
passes data into XGBoost and tells XGBoost whether the iterator has reached its end.
During training, XGBoost will generate some caches for internal data structures in current
directory, which can be changed by `cache_prefix` parameter during construction of
`DMatrix`.
