import itertools
import warnings
from typing import Type

import numpy as np
import pytest
import scipy.sparse

import xgboost as xgb
from xgboost import testing as tm


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "DMatrixT,CSR",
    [
        (m, n)
        for m, n in itertools.product(
            (xgb.DMatrix, xgb.QuantileDMatrix),
            (scipy.sparse.csr_matrix, scipy.sparse.csr_array),
        )
    ],
)
def test_csr(DMatrixT: Type[xgb.DMatrix], CSR: Type) -> None:
    with warnings.catch_warnings():
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = CSR((data, indices, indptr), shape=(3, 3))
        dtrain = DMatrixT(X)
        assert dtrain.num_row() == 3
        assert dtrain.num_col() == 3
        assert dtrain.num_nonmissing() == data.size


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "DMatrixT,CSC",
    [
        (m, n)
        for m, n in itertools.product(
            (xgb.DMatrix, xgb.QuantileDMatrix),
            (scipy.sparse.csc_matrix, scipy.sparse.csc_array),
        )
    ],
)
def test_csc(DMatrixT: Type[xgb.DMatrix], CSC: Type) -> None:
    with warnings.catch_warnings():
        row = np.array([0, 2, 2, 0, 1, 2])
        col = np.array([0, 0, 1, 2, 2, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = CSC((data, (row, col)), shape=(3, 3))
        dtrain = DMatrixT(X)
        assert dtrain.num_row() == 3
        assert dtrain.num_col() == 3
        assert dtrain.num_nonmissing() == data.size

        indptr = np.array([0, 3, 5])
        data = np.array([0, 1, 2, 3, 4])
        row_idx = np.array([0, 1, 2, 0, 2])
        X = CSC((data, row_idx, indptr), shape=(3, 2))
        assert tm.predictor_equal(DMatrixT(X.tocsr()), DMatrixT(X))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "DMatrixT,COO",
    [
        (m, n)
        for m, n in itertools.product(
            (xgb.DMatrix, xgb.QuantileDMatrix),
            (scipy.sparse.coo_matrix, scipy.sparse.coo_array),
        )
    ],
)
def test_coo(DMatrixT: Type[xgb.DMatrix], COO: Type) -> None:
    with warnings.catch_warnings():
        row = np.array([0, 2, 2, 0, 1, 2])
        col = np.array([0, 0, 1, 2, 2, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = COO((data, (row, col)), shape=(3, 3))
        dtrain = DMatrixT(X)
        assert dtrain.num_row() == 3
        assert dtrain.num_col() == 3
        assert dtrain.num_nonmissing() == data.size

        assert tm.predictor_equal(DMatrixT(X.tocsr()), DMatrixT(X))
