from typing import Generator, Sequence

import pytest

from xgboost import testing as tm


@pytest.fixture(scope="session", autouse=True)
def setup_rmm_pool(request, pytestconfig: pytest.Config) -> None:
    tm.setup_rmm_pool(request, pytestconfig)


@pytest.fixture(scope="class")
def local_cuda_client(request, pytestconfig: pytest.Config) -> Generator:
    kwargs = {}
    if hasattr(request, "param"):
        kwargs.update(request.param)
    if pytestconfig.getoption("--use-rmm-pool"):
        if tm.no_rmm()["condition"]:
            raise ImportError("The --use-rmm-pool option requires the RMM package")
        import rmm

        kwargs["rmm_pool_size"] = "2GB"
    if tm.no_dask_cuda()["condition"]:
        raise ImportError("The local_cuda_cluster fixture requires dask_cuda package")
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    yield Client(LocalCUDACluster(**kwargs))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--use-rmm-pool", action="store_true", default=False, help="Use RMM pool"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence) -> None:
    # mark dask tests as `mgpu`.
    mgpu_mark = pytest.mark.mgpu
    for item in items:
        item.add_marker(mgpu_mark)
