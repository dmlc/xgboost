# pylint: disable=too-many-locals, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements
"""XGBoost Training and Cross-Validation Library.

This module provides core training functionality for XGBoost models,
including single model training and k-fold cross-validation with comprehensive
monitoring and callback support.

Main Functions
--------------
train : Train a single XGBoost model with monitoring and callbacks
cv : Perform k-fold cross-validation with various strategies

Key Classes
-----------
CVPack : Container for cross-validation fold data and booster
_PackedBooster : Wrapper for handling multiple boosters in cross-validation

Examples
--------
Basic training:
>>> params = {'objective': 'binary:logistic', 'max_depth': 3}
>>> dtrain = DMatrix(X_train, y_train)
>>> model = train(params, dtrain, num_boost_round=100)

Cross-validation:
>>> cv_results = cv(params, dtrain, num_boost_round=100, nfold=5)

Training with early stopping:
>>> dval = DMatrix(X_val, y_val)
>>> model = train(params, dtrain, num_boost_round=1000,
...               evals=[(dval, 'validation')],
...               early_stopping_rounds=50)
"""
import copy
import os
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
    CallbackContainer,
    EarlyStopping,
    EvaluationMonitor,
    TrainingCallback,
)
from .compat import SKLEARN_INSTALLED, XGBStratifiedKFold
from .core import (
    Booster,
    DMatrix,
    Metric,
    Objective,
    XGBoostError,
    _deprecate_positional_args,
    _RefMixIn,
)

if TYPE_CHECKING:
    from pandas import DataFrame as PdDataFrame

# Module constants
DEFAULT_NUM_BOOST_ROUNDS = 10
DEFAULT_NFOLD = 3
DEFAULT_SEED = 0
MIN_BOOST_ROUNDS = 1
MIN_NFOLD = 2

_CVFolds = Sequence["CVPack"]

_RefError = (
    "Training dataset should be used as a reference when constructing the "
    "`QuantileDMatrix` for evaluation."
)

# Custom exceptions for better error handling
class XGBoostTrainingError(XGBoostError):
    """Exception raised for training-specific errors."""
    pass

class XGBoostValidationError(XGBoostError):
    """Exception raised for input validation errors."""
    pass


def _validate_training_inputs(
    params: Dict[str, Any], 
    dtrain: DMatrix, 
    num_boost_round: int,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None
) -> None:
    """Validate inputs for training functions.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Training parameters dictionary.
    dtrain : DMatrix
        Training dataset.
    num_boost_round : int
        Number of boosting rounds.
    evals : Optional[Sequence[Tuple[DMatrix, str]]], default None
        Evaluation datasets.
        
    Raises
    ------
    XGBoostValidationError
        If any input validation fails.
    TypeError
        If inputs have incorrect types.
    """
    if not isinstance(params, dict):
        raise TypeError(f"Expected dict for params, got {type(params).__name__}")
    
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"Expected DMatrix for dtrain, got {type(dtrain).__name__}")
    
    if not isinstance(num_boost_round, int) or num_boost_round < MIN_BOOST_ROUNDS:
        raise XGBoostValidationError(
            f"num_boost_round must be an integer >= {MIN_BOOST_ROUNDS}, "
            f"got {num_boost_round}"
        )
    
    if evals is not None:
        if not isinstance(evals, (list, tuple)):
            raise TypeError("evals must be a list or tuple of (DMatrix, str) pairs")
        
        for i, eval_pair in enumerate(evals):
            if not isinstance(eval_pair, (list, tuple)) or len(eval_pair) != 2:
                raise XGBoostValidationError(
                    f"Each eval pair must be (DMatrix, str), got {eval_pair} at index {i}"
                )
            
            eval_data, eval_name = eval_pair
            if not isinstance(eval_data, DMatrix):
                raise TypeError(
                    f"Expected DMatrix in eval pair {i}, got {type(eval_data).__name__}"
                )
            
            if not isinstance(eval_name, str):
                raise TypeError(
                    f"Expected str for eval name at index {i}, got {type(eval_name).__name__}"
                )


def _validate_eval_datasets(evals: Sequence[Tuple[DMatrix, str]], dtrain: DMatrix) -> None:
    """Validate evaluation datasets against training data.
    
    Parameters
    ----------
    evals : Sequence[Tuple[DMatrix, str]]
        List of evaluation datasets with names.
    dtrain : DMatrix
        Training dataset for reference validation.
        
    Raises
    ------
    ValueError
        If reference validation fails for QuantileDMatrix.
    """
    for eval_data, _ in evals:
        if (
            isinstance(eval_data, _RefMixIn)
            and eval_data.ref is not weakref.ref(dtrain)
            and eval_data is not dtrain
        ):
            raise ValueError(_RefError)


def _setup_callbacks(
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    verbose_eval: Optional[Union[bool, int]] = True,
    early_stopping_rounds: Optional[int] = None,
    maximize: Optional[bool] = None,
    custom_metric: Optional[Metric] = None,
    obj: Optional[Objective] = None,
    is_cv: bool = False,
    show_stdv: bool = True
) -> CallbackContainer:
    """Set up and configure callback container.
    
    Parameters
    ----------
    callbacks : Optional[Sequence[TrainingCallback]], default None
        User-provided callbacks.
    verbose_eval : Optional[Union[bool, int]], default True
        Verbosity configuration.
    early_stopping_rounds : Optional[int], default None
        Early stopping configuration.
    maximize : Optional[bool], default None
        Whether to maximize metric.
    custom_metric : Optional[Metric], default None
        Custom metric function.
    obj : Optional[Objective], default None
        Custom objective function.
    is_cv : bool, default False
        Whether this is for cross-validation.
    show_stdv : bool, default True
        Whether to show standard deviation in CV.
        
    Returns
    -------
    CallbackContainer
        Configured callback container.
    """
    callback_list = [] if callbacks is None else copy.copy(list(callbacks))
    
    if verbose_eval:
        verbose_eval = 1 if verbose_eval is True else verbose_eval
        callback_list.append(
            EvaluationMonitor(period=verbose_eval, show_stdv=show_stdv)
        )
    
    if early_stopping_rounds:
        callback_list.append(
            EarlyStopping(rounds=early_stopping_rounds, maximize=maximize)
        )
    
    return CallbackContainer(
        callback_list, 
        metric=custom_metric, 
        is_cv=is_cv, 
        output_margin=callable(obj)
    )


def _create_booster_from_datasets(
    params: Dict[str, Any],
    datasets: List[DMatrix],
    model_file: Optional[Union[str, os.PathLike, Booster, bytearray]] = None
) -> Booster:
    """Create booster instance from parameters and datasets.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Booster parameters.
    datasets : List[DMatrix]
        List of datasets (training + evaluation).
    model_file : Optional[Union[str, os.PathLike, Booster, bytearray]], default None
        Pre-trained model to load.
        
    Returns
    -------
    Booster
        Configured booster instance.
    """
    return Booster(params, datasets, model_file=model_file)


def _execute_training_loop(
    bst: Booster,
    cb_container: CallbackContainer,
    start_iteration: int,
    num_boost_round: int,
    dtrain: DMatrix,
    evals: List[Tuple[DMatrix, str]],
    obj: Optional[Objective]
) -> Booster:
    """Execute the main training loop with callbacks.
    
    Parameters
    ----------
    bst : Booster
        Booster instance to train.
    cb_container : CallbackContainer
        Callback container for monitoring.
    start_iteration : int
        Starting iteration number.
    num_boost_round : int
        Total number of boosting rounds.
    dtrain : DMatrix
        Training dataset.
    evals : List[Tuple[DMatrix, str]]
        Evaluation datasets.
    obj : Optional[Objective]
        Custom objective function.
        
    Returns
    -------
    Booster
        Trained booster instance.
    """
    bst = cb_container.before_training(bst)
    
    for i in range(start_iteration, num_boost_round):
        if cb_container.before_iteration(bst, i, dtrain, evals):
            break
        bst.update(dtrain, iteration=i, fobj=obj)
        if cb_container.after_iteration(bst, i, dtrain, evals):
            break
    
    return cb_container.after_training(bst)


@_deprecate_positional_args
def train(
    params: Dict[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUNDS,
    *,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    evals_result: Optional[TrainingCallback.EvalsLog] = None,
    verbose_eval: Optional[Union[bool, int]] = True,
    xgb_model: Optional[Union[str, os.PathLike, Booster, bytearray]] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
) -> Booster:
    """Train an XGBoost model with comprehensive monitoring and callbacks.

    This function trains an XGBoost booster using the provided parameters
    and training data, with support for validation monitoring, early stopping,
    and custom callbacks.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of XGBoost parameters. Common parameters include:
        - 'objective': Learning task objective (e.g., 'reg:squarederror')
        - 'max_depth': Maximum tree depth
        - 'learning_rate': Step size shrinkage
        See XGBoost documentation for complete parameter list.
    dtrain : DMatrix
        Training dataset. Must be a properly constructed DMatrix object
        containing features and labels.
    num_boost_round : int, default 10
        Number of boosting rounds (iterations) to perform. Must be >= 1.
    evals : Optional[Sequence[Tuple[DMatrix, str]]], default None
        List of validation datasets with names for monitoring. Each tuple
        contains (validation_data, dataset_name). Example:
        [(dval, 'validation'), (dtest, 'test')]
    obj : Optional[Objective], default None
        Custom objective function. See XGBoost custom objective tutorial.
    maximize : Optional[bool], default None
        Whether to maximize evaluation metric. If None, determined automatically
        based on metric type.
    early_stopping_rounds : Optional[int], default None
        Activates early stopping. Training stops if validation metric doesn't
        improve for this many consecutive rounds. Requires evals parameter.
    evals_result : Optional[TrainingCallback.EvalsLog], default None
        Dictionary to store evaluation history. Modified in-place during training.
    verbose_eval : Optional[Union[bool, int]], default True
        Controls evaluation printing. If True, prints every round. If int,
        prints every N rounds. If False/None, no printing.
    xgb_model : Optional[Union[str, os.PathLike, Booster, bytearray]], default None
        Pre-trained model to continue training from. Can be file path,
        Booster object, or serialized model bytes.
    callbacks : Optional[Sequence[TrainingCallback]], default None
        Custom callback functions executed during training. Note: callbacks
        are not preserved between training sessions.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function. See XGBoost custom metric tutorial.

    Returns
    -------
    Booster
        Trained XGBoost model ready for prediction and inference.

    Raises
    ------
    TypeError
        If input types are incorrect (e.g., non-DMatrix for dtrain).
    XGBoostValidationError
        If parameter validation fails (e.g., negative num_boost_round).
    ValueError
        If QuantileDMatrix reference validation fails.
    XGBoostError
        If training fails due to XGBoost-specific issues.

    Examples
    --------
    Basic training:
    >>> import xgboost as xgb
    >>> params = {'objective': 'reg:squarederror', 'max_depth': 6}
    >>> dtrain = xgb.DMatrix(X_train, y_train)
    >>> bst = train(params, dtrain, num_boost_round=100)

    Training with validation monitoring:
    >>> dval = xgb.DMatrix(X_val, y_val)
    >>> bst = train(params, dtrain, num_boost_round=100, 
    ...             evals=[(dval, 'validation')])

    Training with early stopping:
    >>> bst = train(params, dtrain, num_boost_round=1000,
    ...             evals=[(dval, 'validation')],
    ...             early_stopping_rounds=50)

    Continuing training from existing model:
    >>> bst_continued = train(params, dtrain, num_boost_round=50,
    ...                      xgb_model=bst)

    Notes
    -----
    - When using early stopping, the returned model is from the last iteration,
      not necessarily the best one. Use model slicing `bst[start:end]` to get
      the best iteration model.
    - Callbacks are executed in the order provided and can modify training.
    - The evals_result parameter is modified in-place to store history.
    - For reproducible results, set random seed in params: {'seed': 42}.
    """
    # Validate all inputs
    _validate_training_inputs(params, dtrain, num_boost_round, evals)
    
    # Process evaluation datasets
    evals = list(evals) if evals else []
    if evals:
        _validate_eval_datasets(evals, dtrain)
    
    # Create booster instance
    all_datasets = [dtrain] + [d[0] for d in evals]
    bst = _create_booster_from_datasets(params, all_datasets, xgb_model)
    
    # Setup callbacks
    cb_container = _setup_callbacks(
        callbacks=callbacks,
        verbose_eval=verbose_eval,
        early_stopping_rounds=early_stopping_rounds,
        maximize=maximize,
        custom_metric=custom_metric,
        obj=obj
    )
    
    # Execute training
    start_iteration = 0
    bst = _execute_training_loop(
        bst, cb_container, start_iteration, num_boost_round, dtrain, evals, obj
    )
    
    # Update evaluation results if provided
    if evals_result is not None:
        evals_result.update(cb_container.history)

    return bst.reset()


class CVPack:
    """Container for cross-validation fold data and booster.
    
    This class encapsulates training and testing datasets for a single
    cross-validation fold, along with the associated booster instance.
    It provides a unified interface for training and evaluation operations
    within the context of cross-validation.
    
    Attributes
    ----------
    dtrain : DMatrix
        Training dataset for this fold.
    dtest : DMatrix
        Testing/validation dataset for this fold.
    watchlist : List[Tuple[DMatrix, str]]
        List of datasets to monitor during training, contains both
        training and testing datasets with their labels.
    bst : Booster
        XGBoost booster instance configured for this fold.
        
    Examples
    --------
    Creating a CV pack:
    >>> dtrain = DMatrix(X_train_fold, y_train_fold)
    >>> dtest = DMatrix(X_test_fold, y_test_fold)
    >>> params = {'objective': 'reg:squarederror'}
    >>> cv_pack = CVPack(dtrain, dtest, params)
    
    Training for one iteration:
    >>> cv_pack.update(0, None)  # iteration 0, no custom objective
    
    Evaluating the fold:
    >>> eval_result = cv_pack.eval(0, None, False)  # iteration 0, no custom metric
    """

    def __init__(
        self, 
        dtrain: DMatrix, 
        dtest: DMatrix, 
        param: Optional[Union[Dict, List]]
    ) -> None:
        """Initialize the CVPack with training and testing data.
        
        Parameters
        ----------
        dtrain : DMatrix
            Training dataset for this cross-validation fold.
        dtest : DMatrix
            Testing dataset for this cross-validation fold.
        param : Optional[Union[Dict, List]]
            Parameters for the booster. Can be dictionary or list of tuples.
            
        Raises
        ------
        TypeError
            If dtrain or dtest are not DMatrix objects.
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"dtrain must be DMatrix, got {type(dtrain).__name__}")
        if not isinstance(dtest, DMatrix):
            raise TypeError(f"dtest must be DMatrix, got {type(dtest).__name__}")
            
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, "train"), (dtest, "test")]
        self.bst = Booster(param, [dtrain, dtest])

    def __getattr__(self, name: str) -> Callable:
        """Delegate attribute access to the underlying booster.
        
        Parameters
        ----------
        name : str
            Attribute name to access from the booster.
            
        Returns
        -------
        Callable
            Function that forwards calls to the booster.
        """
        def _inner(*args: Any, **kwargs: Any) -> Any:
            return getattr(self.bst, name)(*args, **kwargs)
        return _inner

    def update(self, iteration: int, fobj: Optional[Objective]) -> None:
        """Update the booster for one iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        fobj : Optional[Objective]
            Custom objective function to use for this update.
        """
        self.bst.update(self.dtrain, iteration, fobj)

    def eval(self, iteration: int, feval: Optional[Metric], output_margin: bool) -> str:
        """Evaluate the CVPack for one iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        feval : Optional[Metric]
            Custom evaluation metric function.
        output_margin : bool
            Whether to output raw scores (before transformation).
            
        Returns
        -------
        str
            Evaluation result string containing metrics for both train and test sets.
        """
        return self.bst.eval_set(self.watchlist, iteration, feval, output_margin)


class _PackedBooster:
    """Wrapper for handling multiple boosters in cross-validation.
    
    This class provides a unified interface to operate on multiple booster
    instances simultaneously, as required for cross-validation training.
    It forwards operations to all contained boosters and aggregates results
    where appropriate.
    
    Parameters
    ----------
    cvfolds : _CVFolds
        Sequence of CVPack objects containing the individual fold boosters.
        
    Examples
    --------
    >>> cvfolds = [CVPack(dtrain1, dtest1, params), CVPack(dtrain2, dtest2, params)]
    >>> packed = _PackedBooster(cvfolds)
    >>> packed.update(0, None)  # Updates all folds for iteration 0
    >>> results = packed.eval(0, None, False)  # Evaluates all folds
    """
    
    def __init__(self, cvfolds: _CVFolds) -> None:
        """Initialize with cross-validation folds.
        
        Parameters
        ----------
        cvfolds : _CVFolds
            Sequence of CVPack objects for cross-validation.
        """
        self.cvfolds = cvfolds

    def update(self, iteration: int, obj: Optional[Objective]) -> None:
        """Update all folds for the given iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        obj : Optional[Objective]
            Custom objective function applied to all folds.
        """
        for fold in self.cvfolds:
            fold.update(iteration, obj)

    def eval(
        self, iteration: int, feval: Optional[Metric], output_margin: bool
    ) -> List[str]:
        """Evaluate all folds for the given iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        feval : Optional[Metric]
            Custom evaluation metric function.
        output_margin : bool
            Whether to output raw scores.
            
        Returns
        -------
        List[str]
            List of evaluation result strings, one per fold.
        """
        result = [f.eval(iteration, feval, output_margin) for f in self.cvfolds]
        return result

    def set_attr(self, **kwargs: Optional[Any]) -> Any:
        """Set attributes on all fold boosters.
        
        Parameters
        ----------
        **kwargs : Optional[Any]
            Attribute key-value pairs to set on all boosters.
        """
        for f in self.cvfolds:
            f.bst.set_attr(**kwargs)

    def attr(self, key: str) -> Optional[str]:
        """Get attribute from the first fold booster.
        
        Parameters
        ----------
        key : str
            Attribute key to retrieve.
            
        Returns
        -------
        Optional[str]
            Attribute value from the first fold, or None if not found.
        """
        return self.cvfolds[0].bst.attr(key)

    def set_param(
        self,
        params: Union[Dict, Iterable[Tuple[str, Any]], str],
        value: Optional[str] = None,
    ) -> None:
        """Set parameters on all fold boosters.
        
        Parameters
        ----------
        params : Union[Dict, Iterable[Tuple[str, Any]], str]
            Parameters to set. Can be dict, iterable of tuples, or single key.
        value : Optional[str], default None
            Value to set if params is a single key string.
        """
        for f in self.cvfolds:
            f.bst.set_param(params, value)

    def num_boosted_rounds(self) -> int:
        """Get the number of boosted rounds from the first fold.
        
        Returns
        -------
        int
            Number of completed boosting rounds.
        """
        return self.cvfolds[0].num_boosted_rounds()

    @property
    def best_iteration(self) -> int:
        """Get the best iteration from early stopping.
        
        Returns
        -------
        int
            Best iteration number from early stopping.
        """
        return int(cast(int, self.cvfolds[0].bst.attr("best_iteration")))

    @best_iteration.setter
    def best_iteration(self, iteration: int) -> None:
        """Set the best iteration on all folds.
        
        Parameters
        ----------
        iteration : int
            Best iteration number to set.
        """
        self.set_attr(best_iteration=iteration)

    @property
    def best_score(self) -> float:
        """Get the best score from early stopping.
        
        Returns
        -------
        float
            Best score from early stopping.
        """
        return float(cast(float, self.cvfolds[0].bst.attr("best_score")))

    @best_score.setter
    def best_score(self, score: float) -> None:
        """Set the best score on all folds.
        
        Parameters
        ----------
        score : float
            Best score to set.
        """
        self.set_attr(best_score=score)


def groups_to_rows(groups: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Convert group indices to row indices using group boundaries.
    
    This function maps group-level indices to the corresponding row-level
    indices based on predefined group boundaries. It's commonly used in
    ranking problems where data is organized in groups.
    
    Parameters
    ----------
    groups : np.ndarray
        Array of group indices to convert. Should contain valid indices
        within the range [0, len(boundaries)-2].
    boundaries : np.ndarray
        Array defining the start/end boundaries for each group.
        boundaries[i] is the starting row index for group i, and
        boundaries[i+1] is the ending row index (exclusive).
        Length should be number_of_groups + 1.
    
    Returns
    -------
    np.ndarray
        Array of row indices corresponding to all rows in the specified groups.
        
    Examples
    --------
    Convert groups 0 and 2 to row indices:
    >>> groups = np.array([0, 2])
    >>> boundaries = np.array([0, 3, 6, 10])  # 3 groups with sizes [3, 3, 4]
    >>> groups_to_rows(groups, boundaries)
    array([0, 1, 2, 6, 7, 8, 9])
    
    Single group conversion:
    >>> groups = np.array([1])
    >>> boundaries = np.array([0, 5, 10, 15])
    >>> groups_to_rows(groups, boundaries)
    array([5, 6, 7, 8, 9])
        
    Raises
    ------
    IndexError
        If groups contains indices outside the valid range.
    ValueError
        If boundaries array is malformed or too short.
    """
    if len(boundaries) < 2:
        raise ValueError("boundaries array must have at least 2 elements")
    
    max_group_idx = len(boundaries) - 2
    if len(groups) > 0 and (groups.min() < 0 or groups.max() > max_group_idx):
        raise IndexError(
            f"Group indices must be in range [0, {max_group_idx}], "
            f"got range [{groups.min()}, {groups.max()}]"
        )
    
    return np.concatenate([np.arange(boundaries[g], boundaries[g + 1]) for g in groups])


def _validate_cv_inputs(
    params: BoosterParam,
    dtrain: DMatrix,
    num_boost_round: int,
    nfold: int
) -> None:
    """Validate inputs for cross-validation functions.
    
    Parameters
    ----------
    params : BoosterParam
        Training parameters.
    dtrain : DMatrix
        Training dataset.
    num_boost_round : int
        Number of boosting rounds.
    nfold : int
        Number of cross-validation folds.
        
    Raises
    ------
    XGBoostValidationError
        If any validation fails.
    TypeError
        If input types are incorrect.
    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"Expected DMatrix for dtrain, got {type(dtrain).__name__}")
    
    if isinstance(dtrain, _RefMixIn):
        raise ValueError("`QuantileDMatrix` is not yet supported.")
    
    if not isinstance(num_boost_round, int) or num_boost_round < MIN_BOOST_ROUNDS:
        raise XGBoostValidationError(
            f"num_boost_round must be an integer >= {MIN_BOOST_ROUNDS}, "
            f"got {num_boost_round}"
        )
    
    if not isinstance(nfold, int) or nfold < MIN_NFOLD:
        raise XGBoostValidationError(
            f"nfold must be an integer >= {MIN_NFOLD}, got {nfold}"
        )


def mkgroupfold(
    *,
    dall: DMatrix,
    nfold: int,
    param: BoosterParam,
    evals: Sequence[str] = (),
    fpreproc: Optional[FPreProcCallable] = None,
    shuffle: bool = True,
) -> List[CVPack]:
    """Create n-fold cross-validation while maintaining group structure.
    
    This function is specifically designed for ranking problems where
    data is organized in groups and group integrity must be preserved
    across folds.
    
    Parameters
    ----------
    dall : DMatrix
        Complete dataset with group information.
    nfold : int
        Number of cross-validation folds to create.
    param : BoosterParam
        Parameters for booster creation.
    evals : Sequence[str], default ()
        Evaluation metrics to include.
    fpreproc : Optional[FPreProcCallable], default None
        Preprocessing function applied to each fold.
    shuffle : bool, default True
        Whether to shuffle groups before splitting.
        
    Returns
    -------
    List[CVPack]
        List of cross-validation fold packages.
        
    Raises
    ------
    ValueError
        If group information is missing or invalid.
    """
    # Get group information
    group_boundaries = dall.get_uint_info("group_ptr")
    if len(group_boundaries) <= 1:
        raise ValueError("Dataset must contain group information for group-based CV")
    
    group_sizes = np.diff(group_boundaries)
    
    # Create group splits
    if shuffle:
        idx = np.random.permutation(len(group_sizes))
    else:
        idx = np.arange(len(group_sizes))
    
    # Split groups across folds
    out_group_idset = np.array_split(idx, nfold)
    in_group_idset = [
        np.concatenate([out_group_idset[i] for i in range(nfold) if k != i])
        for k in range(nfold)
    ]
    
    # Convert group indices to row indices
    in_idset = [
        groups_to_rows(in_groups, group_boundaries) for in_groups in in_group_idset
    ]
    out_idset = [
        groups_to_rows(out_groups, group_boundaries) for out_groups in out_group_idset
    ]

    # Create CV packs
    ret = []
    for k in range(nfold):
        # Create fold datasets
        dtrain = dall.slice(in_idset[k], allow_groups=True)
        dtrain.set_group(group_sizes[in_group_idset[k]])
        dtest = dall.slice(out_idset[k], allow_groups=True)
        dtest.set_group(group_sizes[out_group_idset[k]])
        
        # Apply preprocessing if provided
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        
        # Create parameter list with evaluation metrics
        plst = list(tparam.items()) + [("eval_metric", itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    
    return ret


def mknfold(
    *,
    dall: DMatrix,
    nfold: int,
    param: BoosterParam,
    seed: int,
    evals: Sequence[str] = (),
    fpreproc: Optional[FPreProcCallable] = None,
    stratified: Optional[bool] = False,
    folds: Optional[XGBStratifiedKFold] = None,
    shuffle: bool = True,
) -> List[CVPack]:
    """Create n-fold cross-validation with various splitting strategies.
    
    This function creates cross-validation folds using different strategies:
    standard k-fold, stratified k-fold, group-based k-fold, or custom folds.
    
    Parameters
    ----------
    dall : DMatrix
        Complete dataset to split into folds.
    nfold : int
        Number of cross-validation folds.
    param : BoosterParam
        Parameters for booster creation.
    seed : int
        Random seed for reproducible splits.
    evals : Sequence[str], default ()
        Evaluation metrics to include.
    fpreproc : Optional[FPreProcCallable], default None
        Preprocessing function applied to each fold.
    stratified : Optional[bool], default False
        Whether to use stratified sampling.
    folds : Optional[XGBStratifiedKFold], default None
        Custom fold specification or sklearn splitter object.
    shuffle : bool, default True
        Whether to shuffle data before splitting.
        
    Returns
    -------
    List[CVPack]
        List of cross-validation fold packages.
        
    Raises
    ------
    ValueError
        If conflicting parameters are provided.
    TypeError
        If folds parameter has incorrect type.
    """
    evals = list(evals)
    np.random.seed(seed)

    if stratified is False and folds is None:
        # Standard k-fold cross validation
        if len(dall.get_uint_info("group_ptr")) > 1:
            # Use group-based folding for grouped data
            return mkgroupfold(
                dall=dall,
                nfold=nfold,
                param=param,
                evals=evals,
                fpreproc=fpreproc,
                shuffle=shuffle,
            )

        # Regular k-fold split
        if shuffle:
            idx = np.random.permutation(dall.num_row())
        else:
            idx = np.arange(dall.num_row())
        
        out_idset = np.array_split(idx, nfold)
        in_idset = [
            np.concatenate([out_idset[i] for i in range(nfold) if k != i])
            for k in range(nfold)
        ]
    elif folds is not None:
        # Use custom folds
        try:
            # Try to extract indices directly
            in_idset = [x[0] for x in folds]
            out_idset = [x[1] for x in folds]
        except (TypeError, IndexError):
            # Assume sklearn-style splitter object
            try:
                splits = list(folds.split(X=dall.get_label(), y=dall.get_label()))
                in_idset = [x[0] for x in splits]
                out_idset = [x[1] for x in splits]
            except AttributeError:
                raise TypeError(
                    "folds must be either a list of (train_idx, test_idx) tuples "
                    "or an sklearn splitter object with a split() method"
                )
        nfold = len(out_idset)
    else:
        # Stratified k-fold split
        if not SKLEARN_INSTALLED:
            raise XGBoostError(
                "sklearn needs to be installed to use stratified cross-validation"
            )
        
        sfk = XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        splits = list(sfk.split(X=dall.get_label(), y=dall.get_label()))
        in_idset = [x[0] for x in splits]
        out_idset = [x[1] for x in splits]
        nfold = len(out_idset)

    # Create CV packs
    ret = []
    for k in range(nfold):
        # Create fold datasets
        dtrain = dall.slice(in_idset[k])
        dtest = dall.slice(out_idset[k])
        
        # Apply preprocessing if provided
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        
        # Create parameter list with evaluation metrics
        plst = list(tparam.items()) + [("eval_metric", itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    
    return ret


def _process_cv_params(params: BoosterParam) -> Tuple[Dict[str, Any], List[str]]:
    """Process and validate cross-validation parameters.
    
    Parameters
    ----------
    params : BoosterParam
        Input parameters (dict or list of tuples).
        
    Returns
    -------
    Tuple[Dict[str, Any], List[str]]
        Processed parameters dictionary and list of metrics.
    """
    params_copy = params.copy()
    
    if isinstance(params_copy, list):
        # Extract metrics from list format
        metrics = [x[1] for x in params_copy if x[0] == "eval_metric"]
        params_dict = dict(params_copy)
        if "eval_metric" in params_dict:
            params_dict["eval_metric"] = metrics
    else:
        params_dict = params_copy
        metrics = []

    # Extract metrics from parameters
    if "eval_metric" in params_dict:
        if isinstance(params_dict["eval_metric"], list):
            metrics = params_dict["eval_metric"]
        else:
            metrics = [params_dict["eval_metric"]]

    # Remove eval_metric from params to avoid duplication
    params_dict.pop("eval_metric", None)
    
    return params_dict, metrics


def _execute_cv_training_loop(
    booster: _PackedBooster,
    callbacks_container: CallbackContainer,
    num_boost_round: int,
    dtrain: DMatrix,
    obj: Optional[Objective]
) -> Dict[str, List[float]]:
    """Execute cross-validation training loop.
    
    Parameters
    ----------
    booster : _PackedBooster
        Packed booster containing all CV folds.
    callbacks_container : CallbackContainer
        Container with configured callbacks.
    num_boost_round : int
        Number of boosting rounds.
    dtrain : DMatrix
        Training dataset (for callback interface).
    obj : Optional[Objective]
        Custom objective function.
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing training history with mean and std for each metric.
    """
    results: Dict[str, List[float]] = {}
    callbacks_container.before_training(booster)

    for i in range(num_boost_round):
        if callbacks_container.before_iteration(booster, i, dtrain, None):
            break
        
        booster.update(i, obj)
        should_break = callbacks_container.after_iteration(booster, i, dtrain, None)
        
        # Collect aggregated results
        res = callbacks_container.aggregated_cv
        for key, mean, std in cast(List[Tuple[str, float, float]], res):
            if key + "-mean" not in results:
                results[key + "-mean"] = []
            if key + "-std" not in results:
                results[key + "-std"] = []
            results[key + "-mean"].append(mean)
            results[key + "-std"].append(std)

        if should_break:
            # Truncate results to best iteration
            for k in results.keys():  # pylint: disable=consider-iterating-dictionary
                results[k] = results[k][: (booster.best_iteration + 1)]
            break
    
    callbacks_container.after_training(booster)
    return results


@_deprecate_positional_args
def cv(
    params: BoosterParam,
    dtrain: DMatrix,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUNDS,
    *,
    nfold: int = DEFAULT_NFOLD,
    stratified: bool = False,
    folds: XGBStratifiedKFold = None,
    metrics: Sequence[str] = (),
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    fpreproc: Optional[FPreProcCallable] = None,
    as_pandas: bool = True,
    verbose_eval: Optional[Union[int, bool]] = None,
    show_stdv: bool = True,
    seed: int = DEFAULT_SEED,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    shuffle: bool = True,
    custom_metric: Optional[Metric] = None,
) -> Union[Dict[str, float], "PdDataFrame"]:
    """Perform cross-validation with comprehensive monitoring and callbacks.
    
    This function performs k-fold cross-validation on the provided dataset,
    supporting various splitting strategies, early stopping, and custom metrics.
    Results include both mean and standard deviation across folds.

    Parameters
    ----------
    params : BoosterParam
        Dictionary or list of XGBoost parameters. Common parameters:
        - 'objective': Learning task objective (e.g., 'binary:logistic')
        - 'max_depth': Maximum tree depth
        - 'learning_rate': Step size shrinkage
    dtrain : DMatrix
        Complete dataset for cross-validation. Must not use external memory.
    num_boost_round : int, default 10
        Number of boosting iterations per fold.
    nfold : int, default 3
        Number of cross-validation folds. Must be >= 2.
    stratified : bool, default False
        Whether to use stratified sampling (requires sklearn).
    folds : XGBStratifiedKFold, default None
        Custom fold specification. Can be:
        - List of (train_indices, test_indices) tuples
        - Sklearn splitter object with split() method
    metrics : Sequence[str], default ()
        Additional evaluation metrics to monitor beyond those in params.
    obj : Optional[Objective], default None
        Custom objective function applied to all folds.
    maximize : Optional[bool], default None
        Whether to maximize evaluation metric for early stopping.
    early_stopping_rounds : Optional[int], default None
        Stop training if CV metric doesn't improve for N consecutive rounds.
        Uses average across all folds for stopping decision.
    fpreproc : Optional[FPreProcCallable], default None
        Preprocessing function: (dtrain, dtest, params) -> (dtrain, dtest, params)
        Applied to each fold independently.
    as_pandas : bool, default True
        Return pandas DataFrame if available, otherwise numpy dict.
    verbose_eval : Optional[Union[int, bool]], default None
        Print progress. If True, print every round. If int, print every N rounds.
    show_stdv : bool, default True
        Show standard deviation in verbose output.
    seed : int, default 0
        Random seed for reproducible fold creation.
    callbacks : Optional[Sequence[TrainingCallback]], default None
        Custom callbacks applied during CV training.
    shuffle : bool, default True
        Shuffle data before creating folds.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function.

    Returns
    -------
    Union[Dict[str, List[float]], PdDataFrame]
        Cross-validation results. Contains columns/keys for each metric
        with both '-mean' and '-std' suffixes. If as_pandas=True and
        pandas is available, returns DataFrame, otherwise dict.

    Raises
    ------
    XGBoostValidationError
        If input validation fails.
    ValueError
        If QuantileDMatrix is provided (not supported).
    XGBoostError
        If stratified=True but sklearn is not installed.
    TypeError
        If input types are incorrect.

    Examples
    --------
    Basic cross-validation:
    >>> import xgboost as xgb
    >>> params = {'objective': 'binary:logistic', 'max_depth': 3}
    >>> dtrain = xgb.DMatrix(X, y)
    >>> cv_results = cv(params, dtrain, num_boost_round=100, nfold=5)
    >>> print(cv_results['test-logloss-mean'].iloc[-1])  # Final CV score

    Cross-validation with early stopping:
    >>> cv_results = cv(params, dtrain, num_boost_round=1000, 
    ...                 nfold=5, early_stopping_rounds=50)
    
    Stratified cross-validation:
    >>> cv_results = cv(params, dtrain, num_boost_round=100,
    ...                 nfold=5, stratified=True)

    Custom fold specification:
    >>> from sklearn.model_selection import StratifiedKFold
    >>> skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> cv_results = cv(params, dtrain, num_boost_round=100, folds=skf)

    With custom preprocessing:
    >>> def preprocess(dtrain, dtest, params):
    ...     # Custom preprocessing logic
    ...     return dtrain, dtest, params
    >>> cv_results = cv(params, dtrain, num_boost_round=100, 
    ...                 fpreproc=preprocess)

    Notes
    -----
    - For ranking problems with groups, group structure is automatically preserved.
    - Early stopping uses the average metric across all folds for decisions.
    - Results always include standard deviation, regardless of show_stdv setting.
    - Custom callbacks are applied to the aggregated cross-validation process.
    - Set seed parameter for reproducible results across runs.
    """
    # Validate inputs
    _validate_cv_inputs(params, dtrain, num_boost_round, nfold)
    
    # Check stratified requirements
    if stratified and not SKLEARN_INSTALLED:
        raise XGBoostError(
            "sklearn needs to be installed in order to use stratified cv"
        )

    # Process parameters and metrics
    params_dict, extracted_metrics = _process_cv_params(params)
    
    # Determine final metrics list
    if isinstance(metrics, str):
        metrics = [metrics]
    
    final_metrics = list(metrics) if metrics else extracted_metrics

    # Create cross-validation folds
    cvfolds = mknfold(
        dall=dtrain,
        nfold=nfold,
        param=params_dict,
        seed=seed,
        evals=final_metrics,
        fpreproc=fpreproc,
        stratified=stratified,
        folds=folds,
        shuffle=shuffle,
    )

    # Setup callbacks
    callbacks_container = _setup_callbacks(
        callbacks=callbacks,
        verbose_eval=verbose_eval,
        early_stopping_rounds=early_stopping_rounds,
        maximize=maximize,
        custom_metric=custom_metric,
        obj=obj,
        is_cv=True,
        show_stdv=show_stdv
    )

    # Execute cross-validation training
    booster = _PackedBooster(cvfolds)
    results = _execute_cv_training_loop(
        booster, callbacks_container, num_boost_round, dtrain, obj
    )
    
    # Convert to pandas if requested and available
    if as_pandas:
        try:
            import pandas as pd
            results = pd.DataFrame.from_dict(results)
        except ImportError:
            pass  # Return dict format if pandas not available

    return results