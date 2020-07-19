import pathlib
import ctypes
import xgboost
from sklearn.datasets import load_boston

libpath = pathlib.Path(__file__).parent.absolute() / 'build' / 'librmm_bridge.so'
bridge = ctypes.cdll.LoadLibrary(libpath)

xgboost.set_gpu_alloc_callback(bridge.allocate, bridge.deallocate)

X, y = load_boston(return_X_y=True)

dtrain = xgboost.DMatrix(X, label=y)

params = {'tree_method': 'gpu_hist', 'max_depth' : 1, 'objective': 'reg:squarederror'}

bst = xgboost.train(params, dtrain, 10, [(dtrain, 'train')])
