from __future__ import print_function
#pylint: skip-file
import sys
sys.path.append("../../tests/python")
import xgboost as xgb
import testing as tm
import numpy as np
import unittest
from sklearn.datasets import make_classification
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) ; sys.stderr.flush()
    print(*args, file=sys.stdout, **kwargs) ; sys.stdout.flush()

eprint("Testing DMatrix creation")

rng = np.random.RandomState(1994)

cols = 31
#rowslist = [100, 5000, 60032, 360032]
rowslist = [10]


class TestGPU(unittest.TestCase):
    def test_dmatrix(self):
        eprint("Starting test of dmatrix creation")
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split


        for rows in rowslist:
            
            eprint("Creating train data rows=%d cols=%d" % (rows,cols))
            X, y = make_classification(rows, n_features=cols, random_state=7)
            rowstest = int(rows*0.2)
            eprint("Creating test data rows=%d cols=%d" % (rowstest,cols))
            # note the new random state.  if chose same as train random state, exact methods can memorize and do very well on test even for random data, while hist cannot                     
            Xtest, ytest = make_classification(rowstest, n_features=cols, random_state=8)
            
            eprint("Starting DMatrix(X,y)")
            ag_dtrain0 = xgb.DMatrix(X,y)
            ag_dtrain1 = xgb.DMatrix(X,y,nthread=0)
            ag_dtrain2 = xgb.DMatrix(X,y,nthread=10)
            eprint("Starting DMatrix(Xtest,ytest)")
            ag_dtest0 = xgb.DMatrix(Xtest,ytest)
            ag_dtest1 = xgb.DMatrix(Xtest,ytest,nthread=0)
            ag_dtest2 = xgb.DMatrix(Xtest,ytest,nthread=10)

            np.testing.assert_array_equal(ag_dtrain1.get_label(),ag_dtrain0.get_label())
            np.testing.assert_array_equal(ag_dtrain2.get_label(),ag_dtrain0.get_label())
            np.testing.assert_array_equal(ag_dtest1.get_label(),ag_dtest0.get_label())
            np.testing.assert_array_equal(ag_dtest2.get_label(),ag_dtest0.get_label())
            
            # need to test X, but no way to go from DMatrix to numpy currently.
