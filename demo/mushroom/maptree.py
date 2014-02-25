#!/usr/bin/python
import sys

def loadnmap( fname ):
    nmap = {}
    for l in open(fname):
        arr = l.split()
        nmap[int(arr[0])] = arr[1].strip()
    return nmap

fo = sys.stdout
nmap = loadnmap( 'featname.txt' )
for l in open( 'dump.txt'):
    idx = l.find('[f')
    if idx == -1:
        fo.write(l)
    else:
        fid = int( l[idx+2:len(l)].split('>')[0])
        rl = l[0:idx]+'['+nmap[fid]+']' + l.split()[1].strip()+'\n'
        fo.write(rl)

