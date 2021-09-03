Defining a Custom Data Iterator to Load Data from External Memory
=================================================================

A simple demo for using custom data iterator with XGBoost.  The feature is still
**experimental** and not ready for production use.  If you are not familiar with C API,
please read its introduction in our tutorials and visit the basic demo first.

Defining Data Iterator
----------------------

In the example, we define a custom data iterator with 2 methods: `reset` and `next`.  The
`next` method passes data into XGBoost and tells XGBoost whether the iterator has reached
its end, and the `reset` method resets iterations. One important detail when using the C
API for data iterator is users need to make sure that the data passed into `next` method
must be kept in memory until the next iteration or `reset` is called.  The external memory
DMatrix is not limited to training, but also valid for other features like prediction.