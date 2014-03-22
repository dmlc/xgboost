#!/usr/bin/python
import sys
import json

def loadnmap( fname ):
    nmap = {}
    for l in open(fname):
        arr = l.split()
        nmap[int(arr[0])] = arr[1].strip()
    return nmap

def recstats( rec, l, label ):
    for it in l.split(','):
        k = int( it )
        if k not in rec:
            rec[ k ] = (0,0)
        else:
            if label == 0:
                rec[k] = (rec[k][0]+1,rec[k][1])
            else:
                rec[k] = (rec[k][0],rec[k][1]+1)

def loadstats( fname, fpath ):
    res = {}
    fp = open( fname )
    for l in open( fpath ):
        label = int( fp.readline().split()[0] )
        arr = l.split()
        for i in xrange( len(arr) ):
            if i not in res:
                res[ i ] = {}
            recstats( res[ i ], arr[i], label )            
    return res

def mapid( idmap, fid, bid ):
    if (bid, fid) not in idmap:
        idmap[ (bid,fid) ] = len(idmap)
    return idmap[ (bid,fid) ]

def dumpjson( fo, trees ):
    fo.write('{\n')
    fo.write('  \"roots\":'+json.dumps( trees['roots'], separators=(' , ',' : ') ) +',\n' )
    fo.write('  \"weights\":'+json.dumps( trees['weights'], separators=(' , ',' : ') ) +',\n' )
    fo.write('  \"nodes\":[\n' )
    fo.write('%s\n   ]' % ',\n'.join(('    %s' % json.dumps( n, separators=(' , ',' : ') ) )   for n in trees['nodes']) )
    fo.write('\n}\n')
        
fo = sys.stdout
nmap = loadnmap( 'featmap.txt' )
stat = loadstats( 'agaricus.txt.test', 'dump.path.txt' )

trees = {'roots':[], 'weights':[], 'nodes':[] }
idmap = {}

for l in open( 'dump.raw.txt'):
    if l.startswith('booster['):
        bid = int( l.split('[')[1].split(']')[0] )
        trees['roots'].append( mapid(idmap,bid,0) )
        trees['weights'].append( 1.0 )
        continue

    node = {}
    rid = int( l.split(':')[0] )
    node['id'] = mapid( idmap, bid, rid )
    node['neg_cnt' ] = stat[ bid ][ rid ][ 0 ]
    node['pos_cnt' ] = stat[ bid ][ rid ][ 1 ] 

    idx = l.find('[f')
    if idx != -1:
        fid = int( l[idx+2:len(l)].split('<')[0])
        node['label'] = nmap[ fid ]
        node['children'] = [ mapid( idmap, bid, int(it.split('=')[1]) ) for it in l.split()[1].split(',') ]
        node['edge_tags'] = ['yes','no']
    else:
        node['label'] = l.split(':')[1].strip()
        node['value'] = float(l.split(':')[1].split('=')[1])

    trees['nodes'].append( node )
trees['nodes'].sort( key = lambda x:x['id'] )
dumpjson( sys.stderr, trees)
