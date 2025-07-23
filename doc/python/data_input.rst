################################
Supported Python data structures
################################

This page is a support matrix for various input types.

.. _py-data:

*******
Markers
*******

- T: Supported.
- F: Not supported.
- NE: Invalid type for the use case. For instance, :py:class:`pandas.Series` can not be multi-target label.
- NPA: Support with the help of numpy array.
- AT: Support with the help of arrow table.
- CPA: Support with the help of cupy array.
- SciCSR: Support with the help of scipy sparse CSR :py:class:`scipy.sparse.csr_matrix`. The conversion to scipy CSR may or may not be possible. Raise a type error if conversion fails.
- FF: We can look forward to having its support in recent future if requested.
- empty: To be filled in.

************
Table Header
************
- `X` means predictor matrix.
- Meta info: label, weight, etc.
- Multi Label: 2-dim label for multi-target.
- Others: Anything else that we don't list here explicitly including formats like `lil`, `dia`, `bsr`. XGBoost will try to convert it into scipy csr.

**************
Support Matrix
**************

+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| Name                    | DMatrix X | QuantileDMatrix X | Sklearn X | Meta Info | Inplace prediction | Multi Label |
+=========================+===========+===================+===========+===========+====================+=============+
| numpy.ndarray           | T         | T                 | T         | T         | T                  | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| scipy.sparse.csr        | T         | T                 | T         | NE        | T                  | F           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| scipy.sparse.csc        | T         | F                 | T         | NE        | F                  | F           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| scipy.sparse.coo        | SciCSR    | F                 | SciCSR    | NE        | F                  | F           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| uri                     | T         | F                 | F         | F         | NE                 | F           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| list                    | NPA       | NPA               | NPA       | NPA       | NPA                | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| tuple                   | NPA       | NPA               | NPA       | NPA       | NPA                | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| pandas.DataFrame        | NPA       | NPA               | NPA       | NPA       | NPA                | NPA         |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| pandas.Series           | NPA       | NPA               | NPA       | NPA       | NPA                | NE          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| cudf.DataFrame          | T         | T                 | T         | T         | T                  | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| cudf.Series             | T         | T                 | T         | T         | FF                 | NE          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| cupy.ndarray            | T         | T                 | T         | T         | T                  | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| torch.Tensor            | T         | T                 | T         | T         | T                  | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| dlpack                  | CPA       | CPA               |           | CPA       | FF                 | FF          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| modin.DataFrame         | NPA       | FF                | NPA       | NPA       | FF                 |             |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| modin.Series            | NPA       | FF                | NPA       | NPA       | FF                 |             |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| pyarrow.Table           | T         | T                 | T         | T         | T                  | T           |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| polars.DataFrame        | AT        | AT                | AT        | AT        | AT                 | AT          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| polars.LazyFrame (WARN) | AT        | AT                | AT        | AT        | AT                 | AT          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| polars.Series           | AT        | AT                | AT        | AT        | AT                 | NE          |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| _\_array\_\_            | NPA       | F                 | NPA       | NPA       | H                  |             |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+
| Others                  | SciCSR    | F                 |           | F         | F                  |             |
+-------------------------+-----------+-------------------+-----------+-----------+--------------------+-------------+

The polars ``LazyFrame.collect`` supports many configurations, ranging from the choice of
query engine to type coercion. XGBoost simply uses the default parameter. Please run
``collect`` to obtain the ``DataFrame`` before passing it into XGBoost for finer control
over the behaviour.