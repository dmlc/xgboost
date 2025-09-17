import pytest

from xgboost import testing as tm


def has_rmm():
    return tm.no_rmm()["condition"]


@pytest.fixture(scope="session", autouse=True)
def setup_rmm_pool(request, pytestconfig):
    tm.setup_rmm_pool(request, pytestconfig)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--use-rmm-pool", action="store_true", default=False, help="Use RMM pool"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--use-rmm-pool"):
        blocklist = [
            "python-gpu/test_gpu_demos.py::test_dask_training",
            "python-gpu/test_gpu_prediction.py::TestGPUPredict::test_shap",
            "python-gpu/test_gpu_linear.py::TestGPULinear",
        ]
        skip_mark = pytest.mark.skip(
            reason="This test is not run when --use-rmm-pool flag is active"
        )
        for item in items:
            if any(item.nodeid.startswith(x) for x in blocklist):
                item.add_marker(skip_mark)
