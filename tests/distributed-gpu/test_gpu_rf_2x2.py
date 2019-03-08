#!/usr/bin/python
import distributed_gpu as dgpu

def params_fun(rank):
    return {
        'n_gpus': 2,
        'gpu_id': 2*rank,
        'tree_method': 'gpu_hist',
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic',
        'subsample': 0.5,
        'colsample_bynode': 0.5,
        'num_parallel_tree': 20
    }

dgpu.run_test('rf.2x2', params_fun, 1)
