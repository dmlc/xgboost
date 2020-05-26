#!/usr/bin/python
import sys
import random

if len(sys.argv) < 2:
    print('Usage: <filename> <k> [nfold = 5]')
    sys.exit(1)

random.seed(10)

k = int(sys.argv[2])
nfold = int(sys.argv[3]) if len(sys.argv) > 3 else 5

with open(sys.argv[1], 'r') as fi, open(sys.argv[1] + '.train', 'w') as ftr, \
        open( sys.argv[1] + '.test', 'w') as fte:
    for l in fi:
        if random.randint(1, nfold) == k:
            print(l, file=fte, end='')
        else:
            print(l, file=ftr, end='')
