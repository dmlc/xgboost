#!/usr/bin/python
# make prediction
import numpy as np
import xgboost as xgb

# path to where the data lies
dpath = 'data'

modelfile = 'higgs.model'
outfile = 'higgs.pred.csv'
# make top 15% as positive
threshold_ratio = 0.15

# load in training data, directly use numpy
dtest = np.loadtxt( dpath+'/test.csv', delimiter=',', skiprows=1 )
data   = dtest[:,1:31]
idx = dtest[:,0]

print ('finish loading from csv ')
xgmat = xgb.DMatrix( data, missing = -999.0 )
bst = xgb.Booster({'nthread':16}, model_file = modelfile)
ypred = bst.predict( xgmat )

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ]

rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1

# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()

print ('finished writing into prediction file')



