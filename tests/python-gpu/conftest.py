import sys
import pytest
import logging

sys.path.append("tests/python")
import testing as tm                          # noqa

def has_rmm():
    try:
        import rmm
        return True
    except ImportError:
        return False

@pytest.fixture(scope='session', autouse=True)
def setup_rmm_pool(request, pytestconfig):
    if pytestconfig.getoption('--use-rmm-pool'):
        if not has_rmm():
            raise ImportError('The --use-rmm-pool option requires the RMM package')
        import rmm
        from dask_cuda.utils import get_n_gpus
        rmm.reinitialize(pool_allocator=True, initial_pool_size=1024*1024*1024,
                         devices=list(range(get_n_gpus())))

@pytest.fixture(scope='function')
def local_cuda_cluster(request, pytestconfig):
    kwargs = {}
    if hasattr(request, 'param'):
        kwargs.update(request.param)
    if pytestconfig.getoption('--use-rmm-pool'):
        if not has_rmm():
            raise ImportError('The --use-rmm-pool option requires the RMM package')
        import rmm
        from dask_cuda.utils import get_n_gpus
        kwargs['rmm_pool_size'] = '2GB'
    if tm.no_dask_cuda()['condition']:
        raise ImportError('The local_cuda_cluster fixture requires dask_cuda package')
    from dask_cuda import LocalCUDACluster
    with LocalCUDACluster(**kwargs) as cluster:
        yield cluster

def pytest_addoption(parser):
    parser.addoption('--use-rmm-pool', action='store_true', default=False, help='Use RMM pool')


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

    # mark dask tests as `mgpu`.
    mgpu_mark = pytest.mark.mgpu
    for item in items:
        if item.nodeid.startswith(
            "python-gpu/test_gpu_with_dask/test_gpu_with_dask.py"
        ) or item.nodeid.startswith(
            "python-gpu/test_gpu_spark/test_gpu_spark.py"
        ):
            item.add_marker(mgpu_mark)
