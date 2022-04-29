"""Shared typing definition."""
import ctypes
import os
import numpy as np
from typing import (
    Optional,
    Any,
    TypeVar,
    Union,
    Sequence,
    Tuple,
    Dict,
    List,
    Callable
)
from typing_extensions import TypedDict, TypeAlias, Literal

# os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/dt.Frame/
# cudf.DataFrame/cupy.array/dlpack
DataType = Any

# xgboost accepts some other possible types in practice due to historical reason, which is
# lesser tested.  For now we encourage users to pass a simple list of string.
FeatureNames = Optional[Sequence[str]]
FeatureTypes = Optional[Sequence[str]]

ArrayLike = Any
PathLike = Union[str, os.PathLike]
CupyT = ArrayLike  # maybe need a stub for cupy arrays
NumpyOrCupy = Any

# ctypes
# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64  # pylint: disable=C0103

CTypeT = Union[
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_uint,
    ctypes.c_size_t,
]

# supported numeric types
CNumeric = Union[
    ctypes.c_float,
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_uint64,
    ctypes.c_int32,
    ctypes.c_int64,
]

# c pointer types
# real type should be, as defined in typeshed
# but this has to be put in a .pyi file
# c_str_ptr_t = ctypes.pointer[ctypes.c_char]
CStrPtr = ctypes.pointer
# c_str_pptr_t = ctypes.pointer[ctypes.c_char_p]
CStrPptr = ctypes.pointer
# c_float_ptr_t = ctypes.pointer[ctypes.c_float]
CFloatPtr = ctypes.pointer

# c_numeric_ptr_t = Union[
#  ctypes.pointer[ctypes.c_float], ctypes.pointer[ctypes.c_double],
#  ctypes.pointer[ctypes.c_uint], ctypes.pointer[ctypes.c_uint64],
#  ctypes.pointer[ctypes.c_int32], ctypes.pointer[ctypes.c_int64]
# ]
CNumericPtr = ctypes.pointer

# template parameter
_T = TypeVar("_T")

class GeneralParams(TypedDict, total=False):
    booster: Literal["gbtree", "gblinear", "dart"]
    verbosity: Literal[0, 1, 2, 3]
    validate_parameters: bool
    nthreads: Optional[int]
    n_jobs: Optional[int]
    disable_default_eval_metric: bool

TreeBoosterParams = TypedDict("TreeBoosterParams", {
        "eta": float,
        "learning_rate": float,
        "gamma": float,
        "min_split_loss": float,
        "max_depth": int,
        "min_child_weight": float,
        "max_delta_step": float,
        "subsample": float,
        "sampling_method": Literal["uniform", "gradient_based"],
        "colsample_bytree": float,
        "colsample_bylevel": float,
        "colsample_bynode": float,
        "lambda": float,
        "reg_lambda": float,
        "alpha": float,
        "reg_alpha": float,
        "tree_method": Literal["auto", "exact", "approx", "hist", "gpu_hist"],
        "sketch_eps": float,
        "scale_pos_weight": float,
        "updater": str,
        "refresh_leaf": Literal[0, 1],
        "process_type": Literal["default", "update"],
        "grow_policy": Literal["depthwise", "lossguide"],
        "max_leaves": int,
        "max_bin": int,
        "predictor": Literal["auto", "cpu_predictor", "gpu_predictor"],
        "num_parallel_tree": int,
        "monotone_constraints": Union[Tuple[Literal[-1, 0, 1], ...], Dict[str, Literal[-1, 0, 1]]],
        "interaction_constraints": List[Union[List[int], List[str]]],
        # Parameters for specific boosters

        # tree_method=hist/gpu_hist/approx
        "single_precision_histogram": bool,
        "max_cat_to_onehot": float,
        # booster=dart
        "sample_type": Literal["uniform", "weighted"],
        "normalize_type": Literal["tree", "forest"],
        "rate_drop": float,
        "one_drop": Literal[0, 1],
        "skip_drop": float,
        # booster=gblinear
        "feature_selector": Literal["cyclic", "shuffle", "random", "greedy", "thrifty"],
        "top_k": int,
    }, total=False)

EvalMetricType: TypeAlias = Union[Literal[
    "rmse",
    "rmsle",
    "mae",
    "mape",
    "mphe",
    "logloss",
    "error",
    "merror",
    "mlogloss",
    "auc",
    "aucpr",
    "ndcg",
    "map",
    "ndcg-",
    "map-",
    "poisson-nloglik",
    "gamma-nloglik",
    "cox-nloglik",
    "gamma-deviance",
    "tweedie-nloglik",
    "aft-nloglik",
    "interval-regression-accuracy"
], str]  # The `str` option is form the evalaluation metrics using `@`

class LearningParameters(TypedDict, total=False):
    objective: Literal[
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:logistic",
        "reg:pseudohubererror",
        "reg:absoluteerror",
        "binary:logistic",
        "binary:logitraw",
        "binary:hinge",
        "count:poisson",
        "survival:cox",
        "survival:aft",
        "multi:softmax",
        "multi:softprob",
        "rank:pairwise",
        "rank:ndcg",
        "rank:map",
        "rank:gamma",
        "reg:gamma",
        "reg:tweedie"
    ]
    obj: Callable[..., Tuple[np.ndarray, np.ndarray]]
    base_score: float
    eval_metric: Union[EvalMetricType, Sequence[EvalMetricType]]
    feval: Callable[..., Tuple[str, float]]
    seed: int
    random_state: int
    seed_per_interation: bool
    # Parameters for specific objectives

    # reg:tweedie
    tweedie_variance_power: float
    # reg:pseudohubererror
    huber_slope: float
    # survival:aft
    aft_loss_distribution: Literal["norma", "logistic", "extreme"]
    aft_loss_distribution_sacle: float
    # multi:softmax
    num_class: int

class Parameters(GeneralParams, TreeBoosterParams, LearningParameters, total=False):
    pass
