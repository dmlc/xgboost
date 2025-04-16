import pytest

from xgboost import testing as tm


class TestPlotting:
    @pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(), tm.no_graphviz()))
    def test_categorical(self) -> None:
        from xgboost.testing.plotting import run_categorical

        run_categorical("hist", "cuda")
