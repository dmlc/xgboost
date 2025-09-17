.. _predict_api:

##########
Prediction
##########

There are a number of prediction functions in XGBoost with various parameters.  This
document attempts to clarify some of confusions around prediction with a focus on the
Python binding, R package is similar when ``strict_shape`` is specified (see below).

******************
Prediction Options
******************

There are a number of different prediction options for the
:py:meth:`xgboost.Booster.predict` method, ranging from ``pred_contribs`` to
``pred_leaf``.  The output shape depends on types of prediction.  Also for multi-class
classification problem, XGBoost builds one tree for each class and the trees for each
class are called a "group" of trees, so output dimension may change due to used model.
After 1.4 release, we added a new parameter called ``strict_shape``, one can set it to
``True`` to indicate a more restricted output is desired.  Assuming you are using
:py:obj:`xgboost.Booster`, here is a list of possible returns:

- When using normal prediction with ``strict_shape`` set to ``True``:

  Output is a 2-dim array with first dimension as rows and second as groups.  For
  regression/survival/ranking/binary classification this is equivalent to a column vector
  with ``shape[1] == 1``.  But for multi-class with ``multi:softprob`` the number of
  columns equals to number of classes.  If strict_shape is set to False then XGBoost might
  output 1 or 2 dim array.

- When using ``output_margin`` to avoid transformation and ``strict_shape`` is set to ``True``:

  Similar to the previous case, output is a 2-dim array, except for that ``multi:softmax``
  has equivalent output shape of ``multi:softprob`` due to dropped transformation.  If
  strict shape is set to False then output can have 1 or 2 dim depending on used model.

- When using ``pred_contribs`` with ``strict_shape`` set to ``True``:

  Output is a 3-dim array, with ``(rows, groups, columns + 1)`` as shape.  Whether
  ``approx_contribs`` is used does not change the output shape. If the strict shape
  parameter is not set, it can be a 2 or 3 dimension array depending on whether
  multi-class model is being used.

- When using ``pred_interactions`` with ``strict_shape`` set to ``True``:

  Output is a 4-dim array, with ``(rows, groups, columns + 1, columns + 1)`` as shape.
  Like the predict contribution case, whether ``approx_contribs`` is used does not change
  the output shape.  If strict shape is set to False, it can have 3 or 4 dims depending on
  the underlying model.

- When using ``pred_leaf`` with ``strict_shape`` set to ``True``:

  Output is a 4-dim array with ``(n_samples, n_iterations, n_classes, n_trees_in_forest)``
  as shape.  ``n_trees_in_forest`` is specified by the ``numb_parallel_tree`` during
  training.  When strict shape is set to False, output is a 2-dim array with last 3 dims
  concatenated into 1.  Also the last dimension is dropped if it equals to 1. When using
  ``apply`` method in scikit learn interface, this is set to False by default.


For R package, when ``strict_shape`` is specified, an ``array`` is returned, with the same
value as Python except R array is column-major while Python numpy array is row-major, so
all the dimensions are reversed.  For example, for a Python ``predict_leaf`` output
obtained by having ``strict_shape=True`` has 4 dimensions: ``(n_samples, n_iterations,
n_classes, n_trees_in_forest)``, while R with ``strict_shape=TRUE`` outputs
``(n_trees_in_forest, n_classes, n_iterations, n_samples)``.

Other than these prediction types, there's also a parameter called ``iteration_range``,
which is similar to model slicing.  But instead of actually splitting up the model into
multiple stacks, it simply returns the prediction formed by the trees within range.
Number of trees created in each iteration equals to :math:`trees_i = num\_class \times
num\_parallel\_tree`.  So if you are training a boosted random forest with size of 4, on
the 3-class classification dataset, and want to use the first 2 iterations of trees for
prediction, you need to provide ``iteration_range=(0, 2)``.  Then the first :math:`2
\times 3 \times 4` trees will be used in this prediction.

**************
Early Stopping
**************

When a model is trained with early stopping, there is an inconsistent behavior between
native Python interface and sklearn/R interfaces.  By default on R and sklearn interfaces,
the ``best_iteration`` is automatically used so prediction comes from the best model.  But
with the native Python interface :py:meth:`xgboost.Booster.predict` and
:py:meth:`xgboost.Booster.inplace_predict` uses the full model.  Users can use
``best_iteration`` attribute with ``iteration_range`` parameter to achieve the same
behavior.  Also the ``save_best`` parameter from :py:obj:`xgboost.callback.EarlyStopping`
might be useful.


***********
Base Margin
***********

There's a training parameter in XGBoost called ``base_score``, and a meta data for
``DMatrix`` called ``base_margin`` (which can be set in ``fit`` method if you are using
scikit-learn interface).  They specifies the global bias for boosted model.  If the latter
is supplied then former is ignored.  ``base_margin`` can be used to train XGBoost model
based on other models.  See demos on boosting from predictions.

*****************
Staged Prediction
*****************

Using the native interface with ``DMatrix``, prediction can be staged (or cached).  For
example, one can first predict on the first 4 trees then run prediction on 8 trees.  After
running the first prediction, result from first 4 trees are cached so when you run the
prediction with 8 trees XGBoost can reuse the result from previous prediction.  The cache
expires automatically upon next prediction, train or evaluation if the cached ``DMatrix``
object is expired (like going out of scope and being collected by garbage collector in
your language environment).

*******************
In-place Prediction
*******************

Traditionally XGBoost accepts only ``DMatrix`` for prediction, with wrappers like
scikit-learn interface the construction happens internally.  We added support for in-place
predict to bypass the construction of ``DMatrix``, which is slow and memory consuming.
The new predict function has limited features but is often sufficient for simple inference
tasks.  It accepts some commonly found data types in Python like :py:obj:`numpy.ndarray`,
:py:obj:`scipy.sparse.csr_matrix` and :py:obj:`cudf.DataFrame` instead of
:py:obj:`xgboost.DMatrix`.  You can call :py:meth:`xgboost.Booster.inplace_predict` to use
it.  Be aware that the output of in-place prediction depends on input data type, when
input is on GPU data output is :py:obj:`cupy.ndarray`, otherwise a :py:obj:`numpy.ndarray`
is returned.

*************
Thread Safety
*************

After 1.4 release, all prediction functions including normal ``predict`` with various
parameters like shap value computation and ``inplace_predict`` are thread safe when
underlying booster is ``gbtree`` or ``dart``, which means as long as tree model is used,
prediction itself should thread safe.  But the safety is only guaranteed with prediction.
If one tries to train a model in one thread and provide prediction at the other using the
same model the behaviour is undefined.  This happens easier than one might expect, for
instance we might accidentally call ``clf.set_params()`` inside a predict function:

.. code-block:: python

    def predict_fn(clf: xgb.XGBClassifier, X):
        X = preprocess(X)
        clf.set_params(n_jobs=1)  # NOT safe!
        return clf.predict_proba(X, iteration_range=(0, 10))

    with ThreadPoolExecutor(max_workers=10) as e:
        e.submit(predict_fn, ...)

*****************************
Privacy-Preserving Prediction
*****************************

`Concrete ML`_ is a third-party open-source library developed by `Zama`_ that proposes gradient
boosting classes similar to ours, but predicting directly over encrypted data, thanks to
Fully Homomorphic Encryption. A simple example would be as follows:

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from concrete.ml.sklearn import XGBClassifier

    x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=10, random_state=42
    )

    # Train in the clear and quantize the weights
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Simulate the predictions in the clear
    y_pred_clear = model.predict(X_test)

    # Compile in FHE
    model.compile(X_train)

    # Generate keys
    model.fhe_circuit.keygen()

    # Run the inference on encrypted inputs!
    y_pred_fhe = model.predict(X_test, fhe="execute")

    print("In clear  :", y_pred_clear)
    print("In FHE    :", y_pred_fhe)
    print(f"Similarity: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")

More information and examples are given in the `Concrete ML documentation`_.

.. _Zama: https://www.zama.ai/
.. _Concrete ML: https://github.com/zama-ai/concrete-ml
.. _Concrete ML documentation: https://docs.zama.ai/concrete-ml
