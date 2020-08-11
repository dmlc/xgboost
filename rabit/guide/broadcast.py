#!/usr/bin/python
"""
demo python script of rabit
"""
from __future__ import print_function
import os
import sys
# add path to wrapper
# for normal run without tracker script, add following line
# sys.path.append(os.path.dirname(__file__) + '/../wrapper')
import rabit

rabit.init()
n = 3
rank = rabit.get_rank()
s = None
if rank == 0:
    s = {'hello world':100, 2:3}
print('@node[%d] before-broadcast: s=\"%s\"' % (rank, str(s)))
s = rabit.broadcast(s, 0)

print('@node[%d] after-broadcast: s=\"%s\"' % (rank, str(s)))
rabit.finalize()
