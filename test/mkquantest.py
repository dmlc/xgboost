#!/usr/bin/python
import math
import sys
import random
import subprocess

funcs = {
    'seq': 'lambda n: sorted([(x,1) for x in range(1,n+1)], key = lambda x:random.random())',
    'seqlogw': 'lambda n: sorted([(x, math.log(x)) for x in range(1,n+1)], key = lambda x:random.random())'
}

if len(sys.argv) < 3:
    print 'Usage: python mkquantest.py <maxn> <eps> [generate-type] [ndata]|./test_quantile'
    print 'Possible generate-types:' 
    for k, v in funcs.items():
        print '\t%s: %s' % (k, v)
    exit(-1)
random.seed(0)
maxn = int(sys.argv[1])
eps = float(sys.argv[2])
if len(sys.argv) > 3:
    method = sys.argv[3]
    assert method in funcs, ('cannot find method %s' % method)
else:
    method = 'seq'
if len(sys.argv) > 4:
    ndata = int(sys.argv[4])
    assert ndata <= maxn, 'ndata must be smaller than maxn'
else:
    ndata = maxn
    
fo = sys.stdout
fo.write('%d\t%g\n' % (maxn, eps))
for x, w in eval(funcs[method])(ndata):
    fo.write(str(x)+'\t'+str(w)+'\n')
