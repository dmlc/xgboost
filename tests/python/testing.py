# coding: utf-8
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED, DT_INSTALLED
from xgboost.compat import CUDF_INSTALLED, DASK_INSTALLED


def no_sklearn():
    return {'condition': not SKLEARN_INSTALLED,
            'reason': 'Scikit-Learn is not installed'}


def no_dask():
    return {'condition': not DASK_INSTALLED,
            'reason': 'Dask is not installed'}


def no_pandas():
    return {'condition': not PANDAS_INSTALLED,
            'reason': 'Pandas is not installed.'}


def no_dt():
    return {'condition': not DT_INSTALLED,
            'reason': 'Datatable is not installed.'}


def no_matplotlib():
    reason = 'Matplotlib is not installed.'
    try:
        import matplotlib.pyplot as _  # noqa
        return {'condition': False,
                'reason': reason}
    except ImportError:
        return {'condition': True,
                'reason': reason}


def no_dask_cuda():
    reason = 'dask_cuda is not installed.'
    try:
        import dask_cuda as _   # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_cudf():
    return {'condition': not CUDF_INSTALLED,
            'reason': 'CUDF is not installed'}


def no_dask_cudf():
    reason = 'dask_cudf is not installed.'
    try:
        import dask_cudf as _   # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}
