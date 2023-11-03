#################################
Consistency for Language Bindings
#################################

XGBoost has many different language bindings developed over the years, some are in the main repository while others live independently. Many features and interfaces are inconsistent with each others, this document aims to provide some guidelines and actionable items for language binding designers.

*******************
Model Serialization
*******************

XGBoost C API exposes a couple functions for serializing a model for persistence storage. These saved files are backward compatible, meaning one can load an older XGBoost model with a newer XGBoost version. If there's change in the model format, we have deprecation notice inside the C++ implementation and public issue for tracking the status. See :doc:`tutorials/saving_model` for details.

As a result, these are considered to be stable and should work across language bindings. For instance, a model trained in R should be fully functioning in C or Python. Please don't pad anything to the output file or buffer.

If there are extra fields that must be saved:

- First review whether the attribute can be retrieved from known properties of the model. For instance, there's a :py:attr:`~xgboost.XGBClassifier.classes_` attribute in the scikit-learn interface :py:class:`~xgboost.XGBClassifier`, which can be obtained through `numpy.arange(n_classes)` and doesn't need to be saved into the model. Preserving version compatibility is not a trivial task and we are still spending a significant amount of time to maintain it. Please don't make complication if it's not necessary.

- Then please consider whether it's universal. For instance, we have added `feature_types` to the model serialization for categorical features (which is a new feature after 1.6), the attribute is useful or will be useful in the future regardless of the language binding.

- If the field is small, we can save it as model attribute (which is a key-value structure). These attributes are ignored by all other language bindings and mostly an ad-hoc storage.

- Lastly, we should use the UBJSON as the default output format when given a chance (not to be burdened by the old binary format).

*********************
Training Continuation
*********************

There are cases where we want to train a model based on the previous model, for boosting trees, it's either adding new trees or modifying the existing trees. This can be normal model update, error recovery, or other special cases we don't know of yet. When it happens, the training iteration should start from 0, not from the lasting boosted rounds of the model. 0 is a special iteration number, we perform some extra checks like whether the label is valid during that iteration. These checks can be expensive but necessary for eliminating silent errors. Keeping the iteration starts from zero allows us to perform these checks only once for each input data.

*********
Inference
*********

The inference function is quite inconsistent among language bindings at the time of writing due to historical reasons, but this makes more important for us to have consistency in mind in the future development.

- Firstly, it's the output shape. There's a relatively new parameter called ``strict_shape`` in XGBoost and is rarely used. We want to make it as the default behavior but couldn't due to compatibility concerns. See :doc:`prediction` for details. In short, if specified, XGBoost C++ implementation can output prediction with the correct shape, instead of letting the language binding to handle it.
- Policy around early stopping is at the moment inconsistent between various interfaces. Some considers the ``best_iteration`` attribute while others don't. We should formalize that all interfaces in the future should use the ``best_iteration`` during inference unless user has explicitly specified the ``iteration_range`` parameter.

****************
Parameter naming
****************

There are many parameter naming conventions other there, XGBoost interfaces try to align with the larger communities. For example, the R package might support parameters naming like ``max.depth=3``, while the Spark package might support ``MaxDepth=3``. These are fine, it's better for the users to keep their pipeline consistent. However, while supporting naming variants, the normal, XGBoost way of naming should also be supported, meaning ``max_depth=3`` should be a valid parameter no-matter what language one is using. If someone were to write duplicated parameter ``max.depth=3, max_depth=3``, a clear error should be preferred instead of prioritizing one over the other.