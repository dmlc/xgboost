import itertools
import json
import os
import shutil
from typing import Optional

import numpy as np
import pytest
from hypothesis import given, note, settings
from scipy.sparse import csr_matrix

import xgboost
from xgboost import testing as tm
from xgboost.testing.data import RelDataCV, simulate_clicks, sort_ltr_samples
from xgboost.testing.params import lambdarank_parameter_strategy
from xgboost.testing.ranking import run_normalization


def test_ndcg_custom_gain():
    def ndcg_gain(y: np.ndarray) -> np.ndarray:
        return np.exp2(y.astype(np.float64)) - 1.0

    X, y, q, w = tm.make_ltr(n_samples=1024, n_features=4, n_query_groups=3, max_rel=3)
    y_gain = ndcg_gain(y)

    byxgb = xgboost.XGBRanker(tree_method="hist", ndcg_exp_gain=True, n_estimators=10)
    byxgb.fit(
        X,
        y,
        qid=q,
        sample_weight=w,
        eval_set=[(X, y)],
        eval_qid=(q,),
        sample_weight_eval_set=(w,),
        verbose=True,
    )
    byxgb_json = json.loads(byxgb.get_booster().save_raw(raw_format="json"))

    bynp = xgboost.XGBRanker(tree_method="hist", ndcg_exp_gain=False, n_estimators=10)
    bynp.fit(
        X,
        y_gain,
        qid=q,
        sample_weight=w,
        eval_set=[(X, y_gain)],
        eval_qid=(q,),
        sample_weight_eval_set=(w,),
        verbose=True,
    )
    bynp_json = json.loads(bynp.get_booster().save_raw(raw_format="json"))

    # Remove the difference in parameter for comparison
    byxgb_json["learner"]["objective"]["lambdarank_param"]["ndcg_exp_gain"] = "0"
    assert byxgb.evals_result() == bynp.evals_result()
    assert byxgb_json == bynp_json

    # test pairwise can handle max_rel > 31, while ndcg metric is using custom gain
    X, y, q, w = tm.make_ltr(n_samples=1024, n_features=4, n_query_groups=3, max_rel=33)
    ranknet = xgboost.XGBRanker(
        tree_method="hist",
        ndcg_exp_gain=False,
        n_estimators=10,
        objective="rank:pairwise",
    )
    ranknet.fit(X, y, qid=q, eval_set=[(X, y)], eval_qid=[q])
    history = ranknet.evals_result()
    assert (
        history["validation_0"]["ndcg@32"][0] < history["validation_0"]["ndcg@32"][-1]
    )


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


def test_error_msg() -> None:
    X, y, qid, w = tm.make_ltr(10, 2, 2, 2)
    ranker = xgboost.XGBRanker()
    with pytest.raises(ValueError, match=r"equal to the number of query groups"):
        ranker.fit(X, y, qid=qid, sample_weight=y)


@given(lambdarank_parameter_strategy)
@settings(deadline=None, print_blob=True)
def test_lambdarank_parameters(params):
    if params["objective"] == "rank:map":
        rel = 1
    else:
        rel = 4
    X, y, q, w = tm.make_ltr(4096, 3, 13, rel)
    ranker = xgboost.XGBRanker(tree_method="hist", n_estimators=64, **params)
    ranker.fit(X, y, qid=q, sample_weight=w, eval_set=[(X, y)], eval_qid=[q])
    for k, v in ranker.evals_result()["validation_0"].items():
        note(v)
        assert v[-1] >= v[0]
        assert ranker.n_features_in_ == 3


@pytest.mark.skipif(**tm.no_pandas())
@pytest.mark.skipif(**tm.no_sklearn())
def test_unbiased() -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X, y, q, w = tm.make_ltr(8192, 2, n_query_groups=6, max_rel=4)
    X, Xe, y, ye, q, qe = train_test_split(X, y, q, test_size=0.2, random_state=3)
    X = csr_matrix(X)
    Xe = csr_matrix(Xe)
    data = RelDataCV((X, y, q), (Xe, ye, qe), max_rel=4)

    train, _ = simulate_clicks(data)
    x, c, y, q = sort_ltr_samples(
        train.X, train.y, train.qid, train.click, train.pos
    )
    df: Optional[pd.DataFrame] = None

    class Position(xgboost.callback.TrainingCallback):
        def after_training(self, model) -> bool:
            nonlocal df
            config = json.loads(model.save_config())
            ti_plus = np.array(config["learner"]["objective"]["ti+"])
            tj_minus = np.array(config["learner"]["objective"]["tj-"])
            df = pd.DataFrame({"ti+": ti_plus, "tj-": tj_minus})
            return model

    ltr = xgboost.XGBRanker(
        n_estimators=8,
        tree_method="hist",
        lambdarank_unbiased=True,
        lambdarank_num_pair_per_sample=12,
        lambdarank_pair_method="topk",
        objective="rank:ndcg",
        callbacks=[Position()],
        boost_from_average=0,
    )
    ltr.fit(x, c, qid=q, eval_set=[(x, c)], eval_qid=[q])

    assert df is not None
    # normalized
    np.testing.assert_allclose(df["ti+"].iloc[0], 1.0)
    np.testing.assert_allclose(df["tj-"].iloc[0], 1.0)
    # less biased on low ranks.
    assert df["ti+"].iloc[-1] < df["ti+"].iloc[0]

    # Training continuation
    ltr.fit(x, c, qid=q, eval_set=[(x, c)], eval_qid=[q], xgb_model=ltr)
    # normalized
    np.testing.assert_allclose(df["ti+"].iloc[0], 1.0)
    np.testing.assert_allclose(df["tj-"].iloc[0], 1.0)


def test_normalization() -> None:
    run_normalization("cpu")


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
