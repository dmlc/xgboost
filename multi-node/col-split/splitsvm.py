#!/usr/bin/python
import sys
import random

# split libsvm file into different subcolumns
if len(sys.argv) < 4:
    print ('Usage:<fin> <fo> k')
    exit(0)

random.seed(10)
fmap = {}

k = int(sys.argv[3])
fi = open( sys.argv[1], 'r' )
fos = []

for i in range(k):
    fos.append(open( sys.argv[2]+'.col%d' % i, 'w' ))
    
for l in open(sys.argv[1]):
    arr = l.split()
    for f in fos:
        f.write(arr[0])
    for it in arr[1:]:
        fid = int(it.split(':')[0])
        if fid not in fmap:
            fmap[fid] = random.randint(0, k-1)
        fos[fmap[fid]].write(' '+it)
    for f in fos:
        f.write('\n')
for f in fos:    
    f.close()
