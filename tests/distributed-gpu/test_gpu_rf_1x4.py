#!/usr/bin/python
import distributed_gpu as dgpu

def params_fun(rank):
    return {
        'n_gpus': 1,
        'gpu_id': rank,
        'tree_method': 'gpu_hist',
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic',
        'subsample': 0.5,
        'colsample_bynode': 0.5
    }

dgpu.run_test('rf.1x4', params_fun)
