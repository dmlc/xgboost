#################################
Consistency for Language Bindings
#################################

XGBoost has many different language bindings developed over the years, some are in the main repository while others live independently. Many features and interfaces are inconsistent with each others, this document aims to provide some guidelines and actionable items for language binding designers.

*******************
Model Serialization
*******************

XGBoost C API exposes a couple functions for serializing a model for persistence storage. These saved files are backward compatible, meaning one can load an older XGBoost model with a newer XGBoost version. If there's change in the model format, we have deprecation notice inside the C++ implementation and public issue for tracking the status. See :doc:`/tutorials/saving_model` for details.

As a result, these are considered to be stable and should work across language bindings. For instance, a model trained in R should be fully functioning in C or Python. Please don't pad anything to the output file or buffer.

If there are extra fields that must be saved:

- First review whether the attribute can be retrieved from known properties of the model. For instance, there's a :py:attr:`~xgboost.XGBClassifier.classes_` attribute in the scikit-learn interface :py:class:`~xgboost.XGBClassifier`, which can be obtained through `numpy.arange(n_classes)` and doesn't need to be saved into the model. Preserving version compatibility is not a trivial task and we are still spending a significant amount of time to maintain it. Please don't make complication if it's not necessary.

- Then please consider whether it's universal. For instance, we have added `feature_types` to the model serialization for categorical features (which is a new feature after 1.6), the attribute is useful or will be useful in the future regardless of the language binding.

- If the field is small, we can save it as model attribute (which is a key-value structure). These attributes are ignored by all other language bindings and mostly an ad-hoc storage.

- Lastly, we should use the UBJSON as the default output format when given a chance (not to be burdened by the old binary format).

*********************
Training Continuation
*********************

There are cases where we want to train a model based on the previous model, for boosting trees, it's either adding new trees or modifying the existing trees. This can be normal model update, error recovery, or other special cases we don't know of yet. When it happens, the training iteration should start from 0, not from the last boosted rounds of the model. 0 is a special iteration number, we perform some extra checks like whether the label is valid during that iteration. These checks can be expensive but necessary for eliminating silent errors. Keeping the iteration starts from zero allows us to perform these checks only once for each input data.

*********
Inference
*********

The inference function is quite inconsistent among language bindings at the time of writing due to historical reasons, but this makes more important for us to have consistency in mind in the future development.

- Firstly, it's the output shape. There's a relatively new parameter called ``strict_shape`` in XGBoost and is rarely used. We want to make it as the default behavior but couldn't due to compatibility concerns. See :doc:`/prediction` for details. In short, if specified, XGBoost C++ implementation can output prediction with the correct shape, instead of letting the language binding to handle it.
- Policy around early stopping is at the moment inconsistent between various interfaces. Some considers the ``best_iteration`` attribute while others don't. We should formalize that all interfaces in the future should use the ``best_iteration`` during inference unless user has explicitly specified the ``iteration_range`` parameter.

****************
Parameter naming
****************

There are many parameter naming conventions out there, Some XGBoost interfaces try to align with the larger communities. For example, the R package might support parameters naming like ``max.depth=3``, while the Spark package might support ``MaxDepth=3``. These are fine, it's better for the users to keep their pipeline consistent. However, while supporting naming variants, the normal, XGBoost way of naming should also be supported, meaning ``max_depth=3`` should be a valid parameter no-matter what language one is using. If someone were to write duplicated parameter ``max.depth=3, max_depth=3``, a clear error should be preferred instead of prioritizing one over the other.

******************
Default Parameters
******************

Like many other machine learning libraries, all parameters from XGBoost can either be inferred from the data or have default values. Bindings should not make copies of these default values and let the XGBoost core decide. When the parameter key is not passed into the C++ core, XGBoost will pick the default accordingly. These defaults are not necessarily optimal, but they are there for consistency. If there's a new choice of default parameter, we can change it inside the core and it will be automatically propagated to all bindings. Given the same set of parameters and data, various bindings should strive to produce the same model. One exception is the `num_boost_rounds`, which exists only in high-level bindings and has various alias like ``n_estimators``. Its default value is close to arbitrary at the moment, we haven't been able to get a good default yet.

*******
Logging
*******

XGBoost has a default logger builtin that can be a wrapper over binding-specific logging facility. For instance, the Python binding registers a callback to use Python :py:mod:`warnings` and :py:func:`print` function to output logging. We want to keep logging native to the larger communities instead of using the ``std::cerr`` from C++.

***********************************
Minimum Amount of Data Manipulation
***********************************

XGBoost is mostly a machine learning library providing boosting algorithm implementation. Some other implementations might perform some sort of data manipulation implicitly like deciding the coding of the data, and transforming the data according to some heuristic before training. We prefer to keep these operations based on necessities instead of convenience to keep the scope of the project well-defined. Whenever possible, we should leave these features to 3-party libraries and consider how a user can compose their pipeline. For instance, XGBoost itself should not perform ordinal encoding for categorical data, users will pick an encoder that fits their use cases (like out-of-core implementation, distributed implementation, known mapping, etc). If some transformations are decided to be part of the algorithm, we can have it inside the core instead of the language binding. Examples would be target-encoding or sketching the response variables. If we were to support them, we could have it inside the core implementation as part of the ML algorithm. This aligns with the same principles of default parameters, various bindings should provide similar (if not the same) results given the same set of parameters and data.

************
Feature Info
************

XGBoost accepts data structures that contain meta info about predictors, including the names and types of features. Example inputs are :py:class:`pandas.DataFrame`, R `data.frame`. We have the following heuristics:
- When the input data structure contains such information, we set the `feature_names` and `feature_types` for `DMatrix` accordingly.
- When a user provides this information as explicit parameters, the user-provided version should override the one provided by the data structure.
- When both sources are missing, the `DMatrix` class contain empty info.