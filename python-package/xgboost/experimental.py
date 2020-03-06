'''Experimental features in XGBoost.  Code in this module are not stable and
may be changed without notice in the future.

'''
from .sklearn import xgboost_model_doc, XGBModel, XGBRegressorBase


@xgboost_model_doc(
    'scikit-learn API for XGBoost multi-target regression.',
    ['estimators', 'model', 'objective'])
class XGBMultiRegressor(XGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    def __init__(self,
                 objective="reg:squarederror",
                 output_type='single',
                 tree_method='exact',
                 **kwargs):
        super().__init__(objective=objective,
                         tree_method=tree_method,
                         output_type=output_type,
                         **kwargs)
