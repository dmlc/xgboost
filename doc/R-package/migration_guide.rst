.. _migation_guide:

Migrating code from previous XGBoost versions
=============================================

XGBoost's R language bindings had large breaking changes between versions 1.x and 2.x. R code that was working with past XGBoost versions might require modifications to work with the newer versions. This guide outlines the main differences:

- Function ``xgboost()``:
    - Previously, this function accepted arguments 'data' and 'label', which have now been renamed to 'x' and 'y', in line with other popular R packages.
    - Previously, the 'data' argument which is now 'x' had to be passed as either an XGBoost 'DMatrix' or as an R matrix. Now the argument allows R data.frames, matrices, and sparse matrices from the 'Matrix' package, but not XGBoost's own DMatrices. Categorical columns will be deduced from the types of the columns when passing a data.frame.
    - Previously, the 'label' data which is now 'y' had to be passed to ``xgboost()`` encoded in the format used by the XGBoost core library - meaning: binary variables had to be encoded to 0/1, bounds for survival objectives had to be passed as different arguments, among others. In the newest versions, 'y' now doesn't need to be manually encoded beforehand: it should be passed as an R object of the corresponding class as regression functions from base R and core R packages for the corresponding XGBoost objective - e.g. classification problems should be passed a ``factor``, survival problems a ``Surv``, regression problems a numeric vector, and so on. Learning-to-rank is not supported by ``xgboost()``, but is supported by ``xgb.train``.
    - Previously, ``xgboost()`` accepted both a ``params`` argument and named arguments under ``...``. Now all training parameters should be passed as named arguments, and all accepted parameters are explicit function arguments with in-package documentation. Some parameters are not allowed as they are determined automatically from the rest of the data, such as the number of classes for multi-classes classification which is determined automatically from 'y'. As well, parameters that have synonyms or which are accepted under different possible arguments (e.g. "eta" and "learning_rate") now accept only their more descriptive form (so "eta" is not accepted, but "learning_rate" is).
    - Models produced by this function ``xgboost()`` are now returned with a different class "xgboost", which is a subclass of "xgb.Booster" but with more metadata and a ``predict`` method with different defaults.
    - This function ``xgboost()`` is now meant for interactive usage only. For package developers who wish to incorporate the XGBoost package, it is highly recommended to use ``xgb.train`` instead, which is a lower-level function that closely mimics the same function from the Python package and is meant to be less subject to breaking changes.

- Function ``xgb.train()``:
    - Previously, ``xgb.train()`` allowed arguments under both a "params" list and as named arguments under ``...``. Now, all training arguments should be passed under ``params``.
    - In order to make it easier to discover and pass parameters, there is now a function ``xgb.params`` which can generate a list to pass to the ``params`` argument. ``xgb.params`` is simply a function with named arguments that lists everything accepted by ``xgb.train`` and offers in-package documentation for all of the arguments, returning a simple named list.
    - Arguments that are meant to be consumed by the DMatrix constructor must be passed directly to ``xgb.DMatrix`` instead (e.g. argument for categorical features or for feature names).
    - Some arguments have been renamed (e.g. previous 'watchlist' is now 'evals', in line with the Python package).
    - The format of the callbacks to pass to ``xgb.train`` has largely been re-written. See the documentation of ``xgb.Callback`` for details.

- Function ``xgb.DMatrix()``:
    - This function now accepts 'data.frame' inputs and determines which features are categorical from their types - anything with type 'factor' or 'character' will be considered as categorical. Note that when passing data to the 'predict' method, the 'factor' variables must have the same encoding (i.e. same levels) as XGBoost will not re-encode them for you.
    - Whereas previously some arguments such as the type of the features had to be passed as a list under argument 'info', they are all now direct function arguments to 'xgb.DMatrix' instead.
    - There are now other varieties of DMatrix constructors that might better fit some uses cases -for example, there is 'xgb.QuantileDMatrix' which will quantize the features straight away (therefore avoiding redundant copies and reducing memory consumption) for the histogram method in XGBoost (but note that quantized DMatrices are not usable with the 'exact' sorted-indices method).
    - Note that data for 'label' still needs to be encoded in the format consumed by the core XGBoost library - e.g. classification objectives should receive 'label' data encoded as zeros and ones.
    - Creation of DMatrices from text files has been deprecated.

- Function ``xgb.cv()``:
    - While previously this function accepted 'data' and 'label' similarly to the old ``xgboost()``, now it accepts only ``xgb.DMatrix`` objects.
    - The function's scope has been expanded to support more functionalities offered by XGBoost, such as survival and learning-to-rank objectives.

- Method ``predict``:
    - There are now two predict methods with different default arguments according to whether the model was produced through ``xgboost()`` or through ``xgb.train()``. Function ``xgboost()`` is more geared towards interactive usage, and thus the defaults for the 'predict' method on such objects (class "xgboost") by default will perform more data validations such as checking that column names match and reordering them otherwise. The 'predict' method for models created through ``xgb.train()`` (class "xgb.Booster") has the same defaults as before, so for example it will not reorder columns to match names under the default behavior.
    - The 'predict' method for objects of class "xgboost" (produced by ``xgboost()``, not by ``xgb.train()``) now can control the types of predictions to make through an argument ``type``, similarly as the 'predict' methods in the 'stats' module of base R - e.g. one can now do ``predict(model, type="class")``; while the 'predict' method for "xgb.Booster" objects (produced by ``xgb.train()``), just like before, controls those through separate arguments such as ``outputmargin``.
    - Previously, predictions using a subset of the trees were using base-0 indexing and range syntax mimicing Python's ranges, whereas now they use base-1 indexing as is common in R, and their behavior for ranges matches that of R's ``seq`` function. Note that the syntax for "use all trees" and "use trees up to early-stopped criteria" have changed (see documentation for details).

- Booster objects:
    - The structure of these objects has been modified - now they are represented as a simple R "ALTLIST" (a special kind of 'list' object) with additional attributes.
    - These objects now cannot be modified by adding more fields to them, but metadata for them can be added as attributes.
    - The objects distinguish between two types of attributes:
        
        - R-side attributes (which can be accessed and modified through R function ``attributes(model)`` and ``attributes(model)$field <- val``), which allow arbitrary objects. Many attributes are automatically added by the model building functions, such as evaluation logs (a ``data.table`` with metrics calculated per iteration), which previously were model fields.
        - C-level attributes, which allow only JSON-compliant data and which can be accessed and set through function ``xgb.attributes(model)``. These C-level attributes are shareable through serialized models in different XGBoost interfaces, while the R-level ones are specific to the R interface. Some attributes that are standard among language bindings of XGBoost, such as the best interation, are kept as C attributes.
    - Previously, models that were just de-serialized from an on-disk format required calling method 'xgb.Booster.complete' on them to finish the full de-serialization process before being usable, or would otherwise call this method on their own automatically automatically at the first call to 'predict'. Serialization is now handled more gracefully, and there are no additional functions/methods involved - i.e. if one saves a model to disk with ``saveRDS()`` and then reads it back with ``readRDS()``, the model will be fully loaded straight away, without needing to call additional methods on it.
