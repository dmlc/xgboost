#!/usr/bin/python

def loadfmap(fname):
    fmap = {}
    nmap = {}
    with open(fname, 'r') as f:
        for l in f:
            arr = l.split()
            if arr[0].find('.') != -1:
                idx = int(arr[0].strip('.'))
                assert idx not in fmap
                fmap[idx] = {}
                ftype = arr[1].strip(':')
                content = arr[2]
            else:
                content = arr[0]
            for it in content.split(','):
                if it.strip() == '':
                    continue
                k, v = it.split('=')
                fmap[idx][v] = len(nmap)
                nmap[len(nmap)] = f'{ftype}={k}'
    return fmap, nmap

def write_nmap(fname, nmap):
    with open(fname, 'w') as f:
        for i in range(len(nmap)):
            print(f'{i}\t{nmap[i]}\ti', file=f)

# start here
fmap, nmap = loadfmap('agaricus-lepiota.fmap')
write_nmap('featmap.txt', nmap)

with open('agaricus-lepiota.data', 'r') as fi, open('agaricus.txt', 'w') as fo:
    for l in fi:
        arr = l.split(',')
        if arr[0] == 'p':
            print('1', file=fo, end='')
        else:
            assert arr[0] == 'e'
            print('0', file=fo, end='')
        for i in range( 1,len(arr) ):
            print(f' {fmap[i][arr[i].strip()]}:1', file=fo, end='')
        print('\n', file=fo, end='')
