# coding: utf-8

import nose

from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED, DT_INSTALLED


def _skip_if_no_sklearn():
    if not SKLEARN_INSTALLED:
        raise nose.SkipTest()


def _skip_if_no_pandas():
    if not PANDAS_INSTALLED:
        raise nose.SkipTest()


def _skip_if_no_dt():
    if not DT_INSTALLED:
        raise nose.SkipTest()


def _skip_if_no_matplotlib():
    try:
        import matplotlib.pyplot as _     # noqa
    except ImportError:
        raise nose.SkipTest()
