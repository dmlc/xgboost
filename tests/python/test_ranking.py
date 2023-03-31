import itertools
import os
import shutil

import numpy as np
from scipy.sparse import csr_matrix

import xgboost
from xgboost import testing as tm


def test_ranking_with_unweighted_data():
    Xrow = np.array([1, 2, 6, 8, 11, 14, 16, 17])
    Xcol = np.array([0, 0, 1, 1,  2,  2,  3,  3])
    X = csr_matrix((np.ones(shape=8), (Xrow, Xcol)), shape=(20, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 1.0, 0.0,
                  0.0, 1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 1.0, 0.0, 0.0])

    group = np.array([5, 5, 5, 5], dtype=np.uint)
    dtrain = xgboost.DMatrix(X, label=y)
    dtrain.set_group(group)

    params = {'eta': 1, 'tree_method': 'exact',
              'objective': 'rank:pairwise', 'eval_metric': ['auc', 'aucpr'],
              'max_depth': 1}
    evals_result = {}
    bst = xgboost.train(params, dtrain, 10, evals=[(dtrain, 'train')],
                        evals_result=evals_result)
    auc_rec = evals_result['train']['auc']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))
    auc_rec = evals_result['train']['aucpr']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))


def test_ranking_with_weighted_data():
    Xrow = np.array([1, 2, 6, 8, 11, 14, 16, 17])
    Xcol = np.array([0, 0, 1, 1,  2,  2,  3,  3])
    X = csr_matrix((np.ones(shape=8), (Xrow, Xcol)), shape=(20, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 1.0, 0.0,
                  0.0, 1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 1.0, 0.0, 0.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    group = np.array([5, 5, 5, 5], dtype=np.uint)
    dtrain = xgboost.DMatrix(X, label=y, weight=weights)
    dtrain.set_group(group)

    params = {'eta': 1, 'tree_method': 'exact',
              'objective': 'rank:pairwise', 'eval_metric': ['auc', 'aucpr'],
              'max_depth': 1}
    evals_result = {}
    bst = xgboost.train(params, dtrain, 10, evals=[(dtrain, 'train')],
                        evals_result=evals_result)
    auc_rec = evals_result['train']['auc']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))
    auc_rec = evals_result['train']['aucpr']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))

    for i in range(1, 11):
        pred = bst.predict(dtrain, iteration_range=(0, i))
        # is_sorted[i]: is i-th group correctly sorted by the ranking predictor?
        is_sorted = []
        for k in range(0, 20, 5):
            ind = np.argsort(-pred[k:k+5])
            z = y[ind+k]
            is_sorted.append(all(i >= j for i, j in zip(z, z[1:])))
        # Since we give weights 1, 2, 3, 4 to the four query groups,
        # the ranking predictor will first try to correctly sort the last query group
        # before correctly sorting other groups.
        assert all(p <= q for p, q in zip(is_sorted, is_sorted[1:]))


class TestRanking:

    @classmethod
    def setup_class(cls):
        """
        Download and setup the test fixtures
        """
        cls.dpath = 'demo/rank/'
        (x_train, y_train, qid_train, x_test, y_test, qid_test,
         x_valid, y_valid, qid_valid) = tm.data.get_mq2008(cls.dpath)

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
        cls.params = {'objective': 'rank:pairwise',
                      'booster': 'gbtree',
                      'eval_metric': ['ndcg']
                      }

    @classmethod
    def teardown_class(cls):
        """
        Cleanup test artifacts from download and unpacking
        :return:
        """
        zip_f = cls.dpath + "MQ2008.zip"
        if os.path.exists(zip_f):
            os.remove(zip_f)
        directory = cls.dpath + "MQ2008"
        if os.path.exists(directory):
            shutil.rmtree(directory)

    def test_training(self):
        """
        Train an XGBoost ranking model
        """
        # specify validations set to watch performance
        watchlist = [(self.dtest, 'eval'), (self.dtrain, 'train')]
        bst = xgboost.train(self.params, self.dtrain, num_boost_round=2500,
                            early_stopping_rounds=10, evals=watchlist)
        assert bst.best_score > 0.98

    def test_cv(self):
        """
        Test cross-validation with a group specified
        """
        cv = xgboost.cv(self.params, self.dtrain, num_boost_round=2500,
                        early_stopping_rounds=10, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert set(cv.keys()) == {
            'test-ndcg-mean', 'train-ndcg-mean', 'test-ndcg-std', 'train-ndcg-std'
        }, "CV results dict key mismatch."

    def test_cv_no_shuffle(self):
        """
        Test cross-validation with a group specified
        """
        cv = xgboost.cv(self.params, self.dtrain, num_boost_round=2500,
                        early_stopping_rounds=10, shuffle=False, nfold=10,
                        as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == 4

    def test_get_group(self):
        """
        Retrieve the group number from the dmatrix
        """
        # test the new getter
        self.dtrain.get_uint_info('group_ptr')

        for d, qid in [(self.dtrain, self.qid_train),
                       (self.dvalid, self.qid_valid),
                       (self.dtest, self.qid_test)]:
            # size of each group
            group_sizes = np.array([len(list(items))
                                    for _key, items in itertools.groupby(qid)])
            # indexes of group boundaries
            group_limits = d.get_uint_info('group_ptr')
            assert len(group_limits) == len(group_sizes)+1
            assert np.array_equal(np.diff(group_limits), group_sizes)
            assert np.array_equal(
                group_sizes, np.diff(d.get_uint_info('group_ptr')))
            assert np.array_equal(group_sizes, np.diff(d.get_uint_info('group_ptr')))
            assert np.array_equal(group_limits, d.get_uint_info('group_ptr'))
