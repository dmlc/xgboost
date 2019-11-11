import numpy as np
from scipy.sparse import csr_matrix
import xgboost
import os
import math
import unittest
import itertools
import shutil
import urllib.request
import zipfile

class TestRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Download and setup the test fixtures
        """
        from sklearn.datasets import load_svmlight_files
        # download the test data
        cls.dpath = 'demo/rank/'
        src = 'https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.zip'
        target = cls.dpath + '/MQ2008.zip'

        if os.path.exists(cls.dpath) and os.path.exists(target):
            print ("Skipping dataset download...")
        else:
            urllib.request.urlretrieve(url=src, filename=target)
            with zipfile.ZipFile(target, 'r') as f:
                f.extractall(path=cls.dpath)

        (x_train, y_train, qid_train, x_test, y_test, qid_test,
         x_valid, y_valid, qid_valid) = load_svmlight_files(
            (cls.dpath + "MQ2008/Fold1/train.txt",
             cls.dpath + "MQ2008/Fold1/test.txt",
             cls.dpath + "MQ2008/Fold1/vali.txt"),
            query_id=True, zero_based=False)
        # instantiate the matrices
        cls.dtrain = xgboost.DMatrix(x_train, y_train)
        cls.dvalid = xgboost.DMatrix(x_valid, y_valid)
        cls.dtest = xgboost.DMatrix(x_test, y_test)
        # set the group counts from the query IDs
        cls.dtrain.set_group([len(list(items))
                              for _key, items in itertools.groupby(qid_train)])
        cls.dtest.set_group([len(list(items))
                             for _key, items in itertools.groupby(qid_test)])
        cls.dvalid.set_group([len(list(items))
                              for _key, items in itertools.groupby(qid_valid)])
        # save the query IDs for testing
        cls.qid_train = qid_train
        cls.qid_test = qid_test
        cls.qid_valid = qid_valid

        # model training parameters
        cls.params = {'booster': 'gbtree',
                      'tree_method': 'gpu_hist',
                      'gpu_id': 0,
                      'predictor': 'gpu_predictor'
                     }
        cls.cpu_params = {'booster': 'gbtree',
                          'tree_method': 'hist',
                          'gpu_id': -1,
                          'predictor': 'cpu_predictor'
                         }

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup test artifacts from download and unpacking
        :return:
        """
        os.remove(cls.dpath + "MQ2008.zip")
        shutil.rmtree(cls.dpath + "MQ2008")

    @classmethod
    def __test_training_with_rank_objective(cls, rank_objective, metric_name, tolerance=1e-02):
        """
        Internal method that trains the dataset using the rank objective on GPU and CPU, evaluates
        the metric and determines if the delta between the metric is within the tolerance level
        :return:
        """
        # specify validations set to watch performance
        watchlist = [(cls.dtest, 'eval'), (cls.dtrain, 'train')]

        num_trees=2500
        check_metric_improvement_rounds=10

        evals_result = {}
        cls.params['objective'] = rank_objective
        cls.params['eval_metric'] = metric_name
        bst = xgboost.train(cls.params, cls.dtrain, num_boost_round=num_trees,
                            early_stopping_rounds=check_metric_improvement_rounds,
                            evals=watchlist, evals_result=evals_result)
        gpu_map_metric = evals_result['train'][metric_name][-1]

        evals_result = {}
        cls.cpu_params['objective'] = rank_objective
        cls.cpu_params['eval_metric'] = metric_name
        bstc = xgboost.train(cls.cpu_params, cls.dtrain, num_boost_round=num_trees,
                             early_stopping_rounds=check_metric_improvement_rounds,
                             evals=watchlist, evals_result=evals_result)
        cpu_map_metric = evals_result['train'][metric_name][-1]

        print("{0} gpu {1} metric {2}".format(rank_objective, metric_name, gpu_map_metric))
        print("{0} cpu {1} metric {2}".format(rank_objective, metric_name, cpu_map_metric))
        print("gpu best score {0} cpu best score {1}".format(bst.best_score, bstc.best_score))
        assert np.allclose(gpu_map_metric, cpu_map_metric, tolerance, tolerance)
        assert np.allclose(bst.best_score, bstc.best_score, tolerance, tolerance)

    def test_training_rank_pairwise_map_metric(self):
        """
        Train an XGBoost ranking model with pairwise objective function and compare map metric
        """
        self.__test_training_with_rank_objective('rank:pairwise', 'map')

    def test_training_rank_pairwise_auc_metric(self):
        """
        Train an XGBoost ranking model with pairwise objective function and compare auc metric
        """
        self.__test_training_with_rank_objective('rank:pairwise', 'auc')

    def test_training_rank_pairwise_ndcg_metric(self):
        """
        Train an XGBoost ranking model with pairwise objective function and compare ndcg metric
        """
        self.__test_training_with_rank_objective('rank:pairwise', 'ndcg')

    def test_training_rank_ndcg_map(self):
        """
        Train an XGBoost ranking model with ndcg objective function and compare map metric
        """
        self.__test_training_with_rank_objective('rank:ndcg', 'map')

    def test_training_rank_ndcg_auc(self):
        """
        Train an XGBoost ranking model with ndcg objective function and compare auc metric
        """
        self.__test_training_with_rank_objective('rank:ndcg', 'auc')

    def test_training_rank_ndcg_ndcg(self):
        """
        Train an XGBoost ranking model with ndcg objective function and compare ndcg metric
        """
        self.__test_training_with_rank_objective('rank:ndcg', 'ndcg')
