#!/usr/bin/python
import sys

fo = open( 'machine.txt', 'w' ) 
cnt = 6
fmap = {}
for l in open( 'machine.data' ):
    arr = l.split(',')
    fo.write(arr[8])
    for i in xrange( 0,6 ):
        fo.write( ' %d:%s' %(i,arr[i+2]) )
    
    if arr[0] not in fmap.keys():
        fmap[arr[0]] = cnt
        cnt += 1
    
    fo.write( ' %d:1' % fmap[arr[0]] )
	
    fo.write('\n')

fo.close()
