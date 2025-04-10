import pytest

from xgboost import testing as tm
from xgboost.testing.plotting import run_categorical

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(), tm.no_graphviz()))


class TestPlotting:
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self) -> None:
        run_categorical("hist", "cuda")
