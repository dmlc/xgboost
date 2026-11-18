# pylint: disable=unused-import
"""Compatibility with optional scikit-learn versions. This is split into an independent
module to avoid importing sklearn if not needed. Sklearn has heavy a dependency tree.

"""

try:
    from sklearn import __version__ as _sklearn_version
    from sklearn.base import BaseEstimator as XGBModelBase
    from sklearn.base import ClassifierMixin as XGBClassifierBase
    from sklearn.base import RegressorMixin as XGBRegressorBase

    # sklearn.utils Tags types can be imported unconditionally once
    # xgboost's minimum scikit-learn version is 1.6 or higher
    try:
        from sklearn.utils import Tags as _sklearn_Tags
    except ImportError:
        _sklearn_Tags = object

    SKLEARN_INSTALLED = True

except ImportError:

    class XGBModelBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.BaseEstimator."""

    class XGBClassifierBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.ClassifierMixin."""

    class XGBRegressorBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.RegressorMixin."""

    _sklearn_Tags = object
    _sklearn_version = object

    SKLEARN_INSTALLED = False
