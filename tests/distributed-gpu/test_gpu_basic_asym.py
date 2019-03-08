#!/usr/bin/python
import distributed_gpu as dgpu

def params_fun(rank):
    return {
        'gpu_id': rank,
        'tree_method': 'gpu_hist',
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic',
        'n_gpus': 1 if rank == 0 else 3
    }

dgpu.run_test('asym', params_fun, 20)
