#!/usr/bin/python
import sys
import random

# split libsvm file into different rows
if len(sys.argv) < 4:
    print ('Usage:<fin> <fo> k')
    exit(0)

random.seed(10)

k = int(sys.argv[3])
fi = open( sys.argv[1], 'r' )
fos = []

for i in range(k):
    fos.append(open( sys.argv[2]+'.row%d' % i, 'w' ))
    
for l in open(sys.argv[1]):
    i = random.randint(0, k-1)
    fos[i].write(l)

for f in fos:    
    f.close()
