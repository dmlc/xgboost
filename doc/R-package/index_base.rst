.. _index_base:

Handling of indexable elements
==============================

There are many functionalities in XGBoost which refer to indexable elements in a countable set, such as boosting rounds / iterations / trees in a model (which can be referred to by number), classes, categories / levels in categorical features, among others.

XGBoost, being written in C++, uses base-0 indexing and considers ranges / sequences to be inclusive of the left end but not the right one - for example, a range (0, 3) would include the first three elements, numbered 0, 1, and 2.

The Python interface uses this same logic, since this is also the way that indexing in Python works, but other languages like R have different logic. In R, indexing is base-1 and ranges / sequences are inclusive of both ends - for example, to refer to the first three elements in a sequence, the interval would be written as (1, 3), and the elements numbered 1, 2, and 3.

In order to provide a more idiomatic R interface, XGBoost adjusts its user-facing R interface to follow this and similar R conventions, but internally, it needs to convert all these numbers to the format that the C interface uses. This is made more problematic by the fact that models are meant to be serializable and loadable in other interfaces, which will have different indexing logic.

The following adjustments are made in the R interface:

- Slicing method for DMatrix, which takes an array of integers, is converted to base-0 indexing by subtracting 1 from each element. Note that this is done in the C-level wrapper function for R, unlike all other conversions which are done in R before being passed to C.
- Slicing method for Booster takes a sequence defined by start, end, and step. The R interface is made to work the same way as R's ``seq`` from the user's POV, so it always adjusts the left end by subtracting one, and depending on whether the step size ends exactly or not at the right end, will also adjust the right end to be non-inclusive in C indexing.
- Parameter ``iterationrange`` in ``predict`` is also made to behave the same way as R's ``seq``. Since it doesn't have a step size, just adjusting the left end by subtracting 1 suffices here.
- ``best_iteration``, depending on the context, might be stored as both a C-level booster attribute, and as an R attribute. Since the C-level attributes are shared across interfaces and used in prediction methods, in order to improve compatibility, it leaves this C-level attribute in base-0 indexing, but the R attribute, if present, will be adjusted to base-1 indexing. Note that the ``predict`` method in R and other interfaces will look at the C-level attribute only.
- Other references to iteration numbers or boosting rounds, such as when printing metrics or saving model snapshots, also follow base-1 indexing. These other references are coded entirely in R, as the C-level functions do not handle such functionalities.
- Terminal leaf / node numbers are returned in base-0 indexing, just like they come from the C interface.
- Tree numbers in plots follow base-1 indexing. Note that these are only displayed when producing these plots through the R interface's own handling of DiagrammeR objects, but not when using the C-level GraphViz 'dot' format generator for plots.
- Feature numbers when producing feature importances, JSONs, trees-to-tables, and SHAP; are all following base-0 indexing.
- Categorical features are defined in R as a ``factor`` type which encodes with base-1 indexing. When categorical features are passed as R ``factor`` types, the conversion is done automatically to base-0 indexing, but if the user whishes to manually supply categorical features as already-encoded integers, then those integers need to already be in base-0 encoding.
- Categorical levels (categories) in outputs such as plots, JSONs, and trees-to-tables; are also referred to using base-0 indexing, regardless of whether they went into the model as integers or as ``factor``-typed columns.
- Categorical labels for DMatrices do not undergo any extra processing - the user must supply base-0 encoded labels.
- A function to retrieve class-specific coefficients when using the linear coefficients history callback takes a class index parameter, which also does not undergo any conversion (i.e. user must pass a base-0 index), in order to match with the label logic - that is, the same class index will refer to the class encoded with that number in the DMatrix ``label`` field.

New additions to the R interface that take on indexable elements should be mindful of these conventions and try to mimic R's behavior as much as possible.
