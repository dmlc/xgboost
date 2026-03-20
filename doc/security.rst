###################
Security disclosure
###################

********************
Use of Python pickle
********************

We use ``pickle`` and ``cloudpickle`` in several places, including a convenient helper function for the ``broadcast`` collective operation to share a Python object. The method is not used internally during training but is here to assist with implementing custom metrics. Also, a distributed interface like PySpark might use pickle to load Python objects, like the callback functions. The security reports state the pickle is unsafe, which is true.

However, from our perspective, if someone else can control your network environment tamper with the data sent between XGBoost workers or the Spark executors, XGBoost should not be the place to provide security mitigation. As for all Python pickles in general, read the warning in the `pickle document <https://docs.python.org/3/library/pickle.html>`__.

Suggestion:

* Don’t train distributed XGBoost models in a public network, or at least don’t do it without a VPN.

***********************************************************
The lack of authentication in the collective implementation
***********************************************************

XGBoost uses TCP sockets for communication between workers during distributed model training. XGBoost is a numeric computation library; the collective module provides high-performance numeric operations (allreduce, allgather, etc.). We will NOT add TLS authentication or encryption into the collective implementation.

Suggestion:

* Don’t train distributed XGBoost models in a public network, or at least don’t do it without a VPN.

***************************************************
The lack of sanitizing for inputs, including models
***************************************************

If someone can manipulate XGBoost inputs, whether with an incorrect model or an altered numpy array, XGBoost will crash due to a memory read error (out-of-bounds access). The reports we received describe manipulating the JSON files to mislead XGBoost into reading overbound values or using conflicting tree indices. We acknowledge that we can add stronger sanitization to the JSON parser when loading from a file. But it’s impractical to check potential issue in the model file at the moment. We rely on mitigation provided by Modern OSs.

Suggestions:

* For most users, this should not cause a significant issue. Your Python program might crash when loading a manipulated JSON model file.
* For hosting XGBoost in a public network environment, follow the usual security practice for loading unknown data.