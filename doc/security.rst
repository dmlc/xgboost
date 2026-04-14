###################
Security disclosure
###################

********************
Use of Python pickle
********************

We use ``pickle`` and ``cloudpickle`` in several places, including a convenient helper function for the ``broadcast`` collective operation to share a Python object. The ``broadcast`` method is not used internally during training but is here to assist with implementing custom metrics. Also, a distributed interface like PySpark might use pickle to transfer Python objects, like the callback functions. Many security scanners will point out the use of pickle as unsafe.

XGBoost as a machine learning library is not designed to protect against pickle data from an untrusted source. Please use appropriate protection mechanisms to ensure that no one can control your network environment and tamper with the pickle data sent between XGBoost workers or the Spark executors. For example, cloud vendors provide managed solutions for running XGBoost in isolated network environments. As for all Python pickles in general, read the warning in the `pickle document <https://docs.python.org/3/library/pickle.html>`__.

Suggestion:

* Do not load pickle files from an unknown source.
* Use secured network for distributed training.

***********************************************************
The lack of authentication in the collective implementation
***********************************************************

XGBoost uses TCP sockets for communication between workers during distributed model training. XGBoost is a numeric computation library; the collective module in intended for high-performance numeric operations (allreduce, allgather, etc.). For performance reasons, we decided that the collective module will NOT support TLS authentication or encryption.

Suggestion:

* Use secured network for distributed training.

***************************************************
The lack of sanitizing for inputs, including models
***************************************************

If someone can manipulate XGBoost inputs, whether with an incorrect model or an altered numpy array, XGBoost will crash due to a memory read error (out-of-bounds access). The reports we received describe manipulating the JSON files to mislead XGBoost into reading out-of-bounds values or using conflicting tree indices. We acknowledge that we can add stronger sanitization to the JSON parser when loading from a file. However, it is currently impractical for us to comprehensively validate all potential issues in a supplied model file. Instead, deployments are expected to rely on standard operating-system–level protections. Examples of non-sanitized inputs:

- Manipulated leaf index in a tree model.
- Manipulated length in a UBJSON model.

Suggestions:

* For most users, this should not cause a security issue. Your Python program might crash when loading a manipulated JSON model file.
* Test the model in an isolated environment before loading it in a critical environment.

***************
Security Policy
***************

==================
Supported Versions
==================

Only the latest XGBoost release is supported.

=========================
Reporting a Vulnerability
=========================

To report a security issue, please email security@xgboost-ci.net with a description of the issue, the steps you took to create the issue, affected versions, and, if known, mitigations for the issue.

All support will be made on the best effort base, so please indicate the "urgency level" of the vulnerability as Critical, High, Medium or Low.
