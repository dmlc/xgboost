# -*- coding: utf-8 -*-
import threading
import unittest
import warnings

import xgboost


class TestRabit(unittest.TestCase):

    def test_rabit_init_twice_ok(self):
        # https://github.com/dmlc/xgboost/issues/2796
        with warnings.catch_warnings(record=True) as w:
            xgboost.rabit.init()
            xgboost.rabit.init()

        assert len(w) == 1
        self.assertIn(str(threading.current_thread().ident), str(w[0]))
