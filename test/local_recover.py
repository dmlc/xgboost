#!/usr/bin/python
from __future__ import print_function
from builtins import range
import rabit
import numpy as np

rabit.init(lib='mock')
rank = rabit.get_rank()
n = 10
nround = 3
data = np.ones(n) * rank

version, model, local = rabit.load_checkpoint(True)
if version == 0:
    model = np.zeros(n)
    local = np.ones(n)
else:
    print('[%d] restart from version %d' % (rank, version))

for i in range(version, nround):
    res = rabit.allreduce(data + model+local, rabit.SUM)
    print('[%d] iter=%d: %s' % (rank, i, str(res)))
    model = res
    local[:] = i
    rabit.checkpoint(model, local)

rabit.finalize()
