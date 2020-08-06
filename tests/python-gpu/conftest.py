import pytest
import logging

def has_rmm():
    try:
        import rmm
        return True
    except ImportError:
        return False

@pytest.fixture(scope='module', autouse=True)
def setup_rmm_pool(request, pytestconfig):
    if pytestconfig.getoption('--use-rmm-pool') and request.module.__name__ != 'test_gpu_with_dask':
        if not has_rmm():
            raise ImportError('The --use-rmm-pool option requires the RMM package')
        import rmm
        from dask_cuda.utils import get_n_gpus
        rmm.reinitialize(pool_allocator=True, devices=list(range(get_n_gpus())))

@pytest.fixture(scope='module', autouse=True)
def local_cuda_cluster_rmm_kwargs(request, pytestconfig):
    if pytestconfig.getoption('--use-rmm-pool') and request.module.__name__ == 'test_gpu_with_dask':
        if not has_rmm():
            raise ImportError('The --use-rmm-pool option requires the RMM package')
        import rmm
        from dask_cuda.utils import get_n_gpus
        rmm.reinitialize()
        return {'rmm_pool_size': '8GB'}
    return {}

def pytest_addoption(parser):
    parser.addoption('--use-rmm-pool', action='store_true', default=False, help='Use RMM pool')
