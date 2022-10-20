import sys

import pytest

from xgboost import testing as tm

sys.path.append("tests/python")
import test_plotting as tp

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(), tm.no_graphviz()))


class TestPlotting:
    cputest = tp.TestPlotting()

    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self):
        self.cputest.run_categorical("gpu_hist")
