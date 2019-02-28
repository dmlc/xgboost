# coding: utf-8
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED, DT_INSTALLED, CUDF_INSTALLED


def no_sklearn():
    return {'condition': not SKLEARN_INSTALLED,
            'reason': 'Scikit-Learn is not installed'}


def no_pandas():
    return {'condition': not PANDAS_INSTALLED,
            'reason': 'Pandas is not installed.'}


def no_dt():
    return {'condition': not DT_INSTALLED,
            'reason': 'Datatable is not installed.'}


def no_cudf():
    return {'condition': not CUDF_INSTALLED,
            'reason': 'cudf is not installed.'}


def no_matplotlib():
    reason = 'Matplotlib is not installed.'
    try:
        import matplotlib.pyplot as _     # noqa
        return {'condition': False,
                'reason': reason}
    except ImportError:
        return {'condition': True,
                'reason': reason}
