import unittest

import pytest
from xgboost import testing as tm

pytestmark = [pytest.mark.skipif(**tm.no_spark())]

from xgboost.spark.utils import _get_default_params_from_func


class UtilsTest(unittest.TestCase):
    def test_get_default_params(self):
        class Foo:
            def func1(self, x, y, key1=None, key2="val2", key3=0, key4=None):
                pass

        unsupported_params = {"key2", "key4"}
        expected_default_params = {
            "key1": None,
            "key3": 0,
        }
        actual_default_params = _get_default_params_from_func(
            Foo.func1, unsupported_params
        )
        self.assertEqual(
            len(expected_default_params.keys()), len(actual_default_params.keys())
        )
        for k, v in actual_default_params.items():
            self.assertEqual(expected_default_params[k], v)
