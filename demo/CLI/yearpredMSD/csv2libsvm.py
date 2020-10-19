#!/usr/bin/python
import sys

if len(sys.argv) < 3:
    print 'Usage: <csv> <libsvm>'
    print 'convert a all numerical csv to libsvm'

fo = open(sys.argv[2], 'w')
for l in open(sys.argv[1]):
    arr = l.split(',')
    fo.write('%s' % arr[0])
    for i in xrange(len(arr) - 1):
        fo.write(' %d:%s' % (i, arr[i+1]))
fo.close()
