#!/usr/bin/python
import distributed_gpu as dgpu

def params_fun(rank):
    return {
        'n_gpus': 4,
        'gpu_id': rank,
        'tree_method': 'gpu_hist',
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic'
    }

dgpu.run_test('4x1', params_fun)
