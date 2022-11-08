#!/usr/bin/env python3

import sys

fo = open(sys.argv[2], 'w')

for l in open(sys.argv[1]):
    arr = l.split(',')
    fo.write('%s' % arr[0])
    for i in range(len(arr) - 1):
        fo.write(' %d:%s' % (i, arr[i+1]))
fo.close()
