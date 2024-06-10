import numpy as np
import xgboost as xgb
from collections import defaultdict
import timeit
import ctypes
from xgboost.core import _LIB, DataSplitMode
from xgboost.data import _check_call, _array_interface, c_bst_ulong, make_jcargs

def measure_create_dmatrix(rows, cols, nthread, use_optimization):
    data =  np.random.randn(rows, cols).astype(np.float32)
    data = np.ascontiguousarray(data)

    handle = ctypes.c_void_p()
    missing = np.nan

    start = timeit.default_timer()
    if use_optimization:
        _LIB.XGDMatrixCreateFromMat_omp(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_bst_ulong(data.shape[0]),
            c_bst_ulong(data.shape[1]),
            ctypes.c_float(missing),
            ctypes.byref(handle),
            ctypes.c_int(nthread),
            ctypes.c_int(DataSplitMode.ROW),
        )
    else:
        _LIB.XGDMatrixCreateFromDense(
            _array_interface(data),
            make_jcargs(
                missing=float(missing),
                nthread=int(nthread),
                data_split_mode=int(DataSplitMode.ROW),
            ),
            ctypes.byref(handle),
        )
    end = timeit.default_timer()
    return end - start

COLS = 1000

print(f"{'Threads':8} | {'Rows':8} | {'Cols':8} | {'Current (sec)':15} | {'Optimized (sec)':15} | {'Ratio':12}")

for nthread in [1, 2, 4, 8]:
    for rows in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
        repeats = 65536 // rows

        current = 0
        for i in range(repeats):
            current += measure_create_dmatrix(rows=rows, cols=COLS, nthread=nthread, use_optimization=False)

        optimized = 0
        for i in range(repeats):
            optimized += measure_create_dmatrix(rows=rows, cols=COLS, nthread=nthread, use_optimization=True)

        print(f"{nthread:8} | {rows:8} | {COLS:8} | {current/repeats:15.4g} | {optimized/repeats:15.4g} | {optimized / current:12.1%}")
