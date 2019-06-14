"""Unit tests for AutoML Library functions."""
import unittest
import automl_core
import xgboost as xgb

class TestAutomlCore(unittest.TestCase):
    """
    A class providing tests for automl_core main functions
    """

    '''
    def test_default_values(self):
        """
        A test checking the setting up of default values for hyper-parameters
        """
        params = {'objective': 'binary:logistic'}
        expected_params = {'objective': 'binary:logistic', 'max_depth': 6, \
                           'learning_rate': 0.3, 'n_estimators': 100, \
                           'eval_metric': 'auc'}
        with self.assertWarns(Warning):
            params = automl_core.xgb_parameter_checker(params, 2)
        self.assertEqual(params, expected_params)
    '''

    def test_objective(self):
        """
        A test for the objective
        """
        params = {}
        with self.assertRaises(automl_core.ParamError):
            automl_core.xgb_parameter_checker(params, 2)

    def test_num_class(self):
        """
        A test for number of classes
        """
        params = {'objective': 'binary:logistic'}
        with self.assertRaises(automl_core.ParamError):
            automl_core.xgb_parameter_checker(params, 4)

    def test_metric(self):
        """
        A test for evaluation metric
        """
        params = {'objective': 'binary:logistic', 'max_depth': 6, \
                  'learning_rate': 0.3, 'n_estimators': 100, \
                  'eval_metric': 'ndcg@ab'}
        with self.assertRaises(automl_core.ParamError):
            params = automl_core.xgb_parameter_checker(params, 2)

    def test_metric_optimization_direction(self):
        """
        A test for metric optimization direction
        """
        params = {'objective': 'binary:logistic', 'max_depth': 6, \
                  'learning_rate': 0.3, 'n_estimators': 100, \
                  'eval_metric': 'auc'}
        automl_core.xgb_parameter_checker(params)
        maximize_eval_metric = params['maximize_eval_metric'].lower() == 'true'
        self.assertTrue(maximize_eval_metric)

if __name__ == '__main__':
    unittest.main()
