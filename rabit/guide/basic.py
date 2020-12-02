#!/usr/bin/python
"""
demo python script of rabit
"""
from __future__ import print_function
from builtins import range
import os
import sys
import numpy as np
# import rabit, the tracker script will setup the lib path correctly
# for normal run without tracker script, add following line
# sys.path.append(os.path.dirname(__file__) + '/../python')
import rabit

rabit.init()
n = 3
rank = rabit.get_rank()
a = np.zeros(n)
for i in range(n):
    a[i] = rank + i

print('@node[%d] before-allreduce: a=%s' % (rank, str(a)))
a = rabit.allreduce(a, rabit.MAX)
print('@node[%d] after-allreduce-max: a=%s' % (rank, str(a)))
a = rabit.allreduce(a, rabit.SUM)
print('@node[%d] after-allreduce-sum: a=%s' % (rank, str(a)))
rabit.finalize()
