###################
Security disclosure
###################

********************
Use of Python pickle
********************

We use ``pickle`` and ``cloudpickle`` in several places, including a convenient helper function for the ``broadcast`` collective operation to share a Python object. The method is not used internally during training but is here to assist with implementing custom metrics. Also, a distributed interface like PySpark might use pickle to load Python objects, like the callback functions. The security reports state the pickle is unsafe, which is true.

XGBoost as a machine learning library is not designed to protect against pickle data from an untrusted source. Please use appropriate protection mechanisms to ensure that no one can control your network environment and tamper with the pickle data sent between XGBoost workers or the Spark executors. For example, cloud vendors provide managed solutions for running XGBoost in an isolated network environments. As for all Python pickles in general, read the warning in the `pickle document <https://docs.python.org/3/library/pickle.html>`__.

Suggestion:

* Don’t train distributed XGBoost models in a public network, or at least don’t do it without a VPN.

***********************************************************
The lack of authentication in the collective implementation
***********************************************************

XGBoost uses TCP sockets for communication between workers during distributed model training. XGBoost is a numeric computation library; the collective module in intended for high-performance numeric operations (allreduce, allgather, etc.). For performance reasons, we decided that the collective module will NOT support TLS authentication or encryption.

Suggestion:

* Don’t train distributed XGBoost models in a public network, or at least don’t do it without a VPN.

***************************************************
The lack of sanitizing for inputs, including models
***************************************************

If someone can manipulate XGBoost inputs, whether with an incorrect model or an altered numpy array, XGBoost will crash due to a memory read error (out-of-bounds access). The reports we received describe manipulating the JSON files to mislead XGBoost into reading out-of-bounds values or using conflicting tree indices. We acknowledge that we can add stronger sanitization to the JSON parser when loading from a file. However, it is currently impractical for us to comprehensively validate all potential issues in a supplied model file. Instead, deployments are expected to rely on standard operating-system–level protections,

Suggestions:

* For most users, this should not cause a significant issue. Your Python program might crash when loading a manipulated JSON model file.
* For hosting XGBoost in a public network environment, follow the usual security practice for loading unknown data.
