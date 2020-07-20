import pathlib
import ctypes
import xgboost
from sklearn.datasets import load_boston

def demo():
    libpath = pathlib.Path(__file__).parent.absolute() / 'build' / 'librmm_bridge.so'
    xgboost.set_gpu_alloc_callback(str(libpath))

    X, y = load_boston(return_X_y=True)
    dtrain = xgboost.DMatrix(X, label=y)
    params = {'tree_method': 'gpu_hist', 'max_depth' : 1, 'objective': 'reg:squarederror'}
    bst = xgboost.train(params, dtrain, 10, [(dtrain, 'train')])

if __name__ == '__main__':
    demo()
