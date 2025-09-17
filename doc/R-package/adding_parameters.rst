Developer guide: parameters from core library
=============================================

The XGBoost core library accepts a long list of input parameters (e.g. ``max_depth`` for decision trees, regularization, ``device`` where compute happens, etc.). New parameters are constantly being added as XGBoost is developed further, and their language bindings should allow passing to the core library everything that it accepts.

In the case of R, these parameters are passed as an R ``list`` object to function ``xgb.train``, but the R interface aims at providing a better, more idiomatic user experience by offering a parameters constructor with full in-package documentation. This requires keeping the list of parameters and their documentation up to date **in the R package** too, in addition to the general online documentation for XGBoost.

In more detail, there is a function ``xgb.params`` which allows the user to construct such a ``list`` object to pass to ``xgb.train`` while getting full IDE autocompletion on it. This function should accept all possible XGBoost parameters as arguments, listing them in the same order as they appear in the online documentation.

In order to add a new parameter from the core library to ``xgb.params``:

- Add the parameter at the right location, according to the order in which it appears in the .rst file listing the parameters for the core library. If the parameter appears more than once (e.g. because it applies to more than one type of booster), then add it in a position according to to the first occurrence.
- Copy-paste the docs from the .rst file as another ``@param`` entry for ``xgb.train``. Some easy substitutions might be needed, such as changing double-backticks to single-backticks, enquoting variables that need to be passed as strings, and replacing ``:math:`` calls with their roxygen equivalent ``\eqn{}``, among others.
- If needed, make minimal modifications for the R interface - for example, since parameters are only listed once, should add at the beginning a note about which type of booster they apply to if they are only applicable for one type, or list default values by booster type if they are different.

After adding the parameter to ``xgb.params``, it will also need to be added to the function ``xgboost`` if that function can use it. The function ``xgboost`` is not meant to support everything that the core library offers - currently parameters related to learning-to-rank are not listed there for example as they are unusable for it (but can be used for ``xgb.train``).

In order to add the parameter to ``xgboost``:

- Add it to the function signature. The position here differs though: there are a few selected parameters whose positions have been moved closer to the top of the signature. New parameters should not be placed within those "top" positions - instead, place it after parameter ``tree_method``, in the most similar place among the remaining parameters according to how it was inserted in ``xgb.params``. Note that the rest of the parameters that come after ``tree_method`` are still meant to follow the same relative order as in ``xgb.params``.
- If the parameter applies exactly in the same way as in ``xgb.train``, then no additional documentation is needed for ``xgboost``, because it inherits parameters from ``xgb.params`` by default. However, some parameters might need slight modifications - for example, not all objectives are supported by ``xgboost``, so modifications are needed for that parameter.
- If the parameter allows aliases, use only one alias, and prefer the most descriptive nomenclature (e.g. "learning_rate" instead of "eta"). These also need a doc entry ``@param`` in ``xgboost``, as the one in ``xgb.params`` will have the unsupported alias.

As new objectives and evaluation metrics are added, be mindful that they need to be added to the docs of both ``xgb.params`` and ``xgboost``. Documentation for objectives in both functions was originally copied from the same .rst file for the core library, but for ``xgboost`` it undergoes additional modifications in order to list what is and isn't supported, and to refer only to the parameter aliases that are accepted by ``xgboost``.

Keep in mind also that objectives that are a variant of one another but with a different prediction mode, are not meant to be allowed in ``xgboost`` as they'd break its intended interface - therefore, such objectives are not described in the docs for ``xgboost`` (but there is a list at the end of what isn't supported by it) and are checked against in function ``prescreen.objective``.
