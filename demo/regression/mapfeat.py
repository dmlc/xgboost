#!/usr/bin/python

with open('machine.data', 'r') as fi, open('machine.txt', 'w') as fo:
    cnt = 6
    fmap = {}
    for l in fi:
        arr = l.split(',')
        print(arr[8], file=fo, end='')
        for i in range(6):
            print(f' {i:d}:{arr[i + 2]:s}', file=fo, end='')

        if arr[0] not in fmap:
            fmap[arr[0]] = cnt
            cnt += 1

        print(f' {fmap[arr[0]]:d}:1', file=fo)

# create feature map for machine data
with open('featmap.txt', 'w') as fo:
    # list from machine.names
    names = ['vendor', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

    for i in range(6):
        print(f'{i}\t{names[i+1]}\tint', file=fo)

    for v, k in sorted(fmap.items(), key=lambda x: x[1]):
        print(f'{k}\tvendor={v}\ti', file=fo)
