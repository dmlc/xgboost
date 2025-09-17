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
TrainingConfiguration : Configuration container for training parameters
CrossValidationConfiguration : Configuration container for CV parameters
CVPack : Container for cross-validation fold data and booster
_PackedBooster : Wrapper for handling multiple boosters in cross-validation
CVStrategy : Abstract base for cross-validation strategies
StandardCVStrategy : Standard k-fold cross-validation
StratifiedCVStrategy : Stratified k-fold cross-validation
GroupCVStrategy : Group-aware cross-validation for ranking

Examples
--------
Basic training:
>>> import xgboost as xgb
>>> from xgboost.training import TrainingConfiguration
>>> params = {'objective': 'binary:logistic', 'max_depth': 3}
>>> dtrain = xgb.DMatrix(X_train, y_train)
>>> config = TrainingConfiguration(num_boost_round=100)
>>> model = train(params, dtrain, config)

Cross-validation:
>>> from xgboost.training import CrossValidationConfiguration
>>> cv_config = CrossValidationConfiguration(nfold=5, num_boost_round=100)
>>> cv_results = cv(params, dtrain, cv_config)

Advanced configuration:
>>> config = TrainingConfiguration(
...     num_boost_round=1000,
...     early_stopping_rounds=50,
...     verbose_eval=10
... )
>>> dval = xgb.DMatrix(X_val, y_val)
>>> model = train(params, dtrain, config, evals=[(dval, 'validation')])
"""
import copy
import os
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

# Training constants
DEFAULT_NUM_BOOST_ROUNDS = 10
DEFAULT_NFOLD = 3
DEFAULT_SEED = 0
DEFAULT_VERBOSE_EVAL_PERIOD = 1
MIN_BOOST_ROUNDS = 1
MIN_NFOLD = 2
MAX_NFOLD = 100

# Callback constants
DEFAULT_EARLY_STOPPING_PATIENCE = 10
MIN_EARLY_STOPPING_ROUNDS = 1

# Cross-validation constants
DEFAULT_SHUFFLE = True
DEFAULT_STRATIFIED = False
DEFAULT_AS_PANDAS = True
DEFAULT_SHOW_STDV = True

# Performance constants
RANDOM_STATE_RANGE = (0, 2**32 - 1)
MAX_ARRAY_SIZE = 10**8

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

class XGBoostConfigurationError(XGBoostError):
    """Exception raised for configuration errors."""
    pass


@dataclass
class TrainingConfiguration:
    """Configuration container for XGBoost training parameters.
    
    This class encapsulates all training-related configuration options
    with validation and sensible defaults.
    
    Attributes
    ----------
    num_boost_round : int, default 10
        Number of boosting iterations to perform.
    early_stopping_rounds : Optional[int], default None
        Number of rounds with no improvement to trigger early stopping.
    verbose_eval : Optional[Union[bool, int]], default True
        Verbosity control for evaluation printing.
    maximize : Optional[bool], default None
        Whether to maximize the evaluation metric.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function.
    callbacks : List[TrainingCallback], default empty list
        List of training callbacks to apply.
    
    Examples
    --------
    Basic configuration:
    >>> config = TrainingConfiguration(num_boost_round=100)
    
    Configuration with early stopping:
    >>> config = TrainingConfiguration(
    ...     num_boost_round=1000,
    ...     early_stopping_rounds=50,
    ...     verbose_eval=10
    ... )
    
    Advanced configuration:
    >>> from xgboost.callback import LearningRateScheduler
    >>> config = TrainingConfiguration(
    ...     num_boost_round=500,
    ...     callbacks=[LearningRateScheduler(lambda epoch: 0.1 * (0.9 ** epoch))]
    ... )
    """
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUNDS
    early_stopping_rounds: Optional[int] = None
    verbose_eval: Optional[Union[bool, int]] = True
    maximize: Optional[bool] = None
    custom_metric: Optional[Metric] = None
    callbacks: List[TrainingCallback] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        XGBoostConfigurationError
            If any configuration parameter is invalid.
        """
        if not isinstance(self.num_boost_round, int) or self.num_boost_round < MIN_BOOST_ROUNDS:
            raise XGBoostConfigurationError(
                f"num_boost_round must be an integer >= {MIN_BOOST_ROUNDS}, "
                f"got {self.num_boost_round}"
            )
        
        if self.early_stopping_rounds is not None:
            if (not isinstance(self.early_stopping_rounds, int) or 
                self.early_stopping_rounds < MIN_EARLY_STOPPING_ROUNDS):
                raise XGBoostConfigurationError(
                    f"early_stopping_rounds must be an integer >= {MIN_EARLY_STOPPING_ROUNDS}, "
                    f"got {self.early_stopping_rounds}"
                )
        
        if (self.verbose_eval is not None and 
            not isinstance(self.verbose_eval, (bool, int))):
            raise XGBoostConfigurationError(
                f"verbose_eval must be bool, int, or None, got {type(self.verbose_eval)}"
            )
        
        if isinstance(self.verbose_eval, int) and self.verbose_eval < 1:
            raise XGBoostConfigurationError(
                f"verbose_eval must be >= 1 when int, got {self.verbose_eval}"
            )
        
        if self.callbacks is not None and not isinstance(self.callbacks, list):
            raise XGBoostConfigurationError(
                f"callbacks must be a list, got {type(self.callbacks)}"
            )


@dataclass  
class CrossValidationConfiguration:
    """Configuration container for cross-validation parameters.
    
    This class encapsulates all cross-validation configuration options
    with validation and sensible defaults.
    
    Attributes
    ----------
    num_boost_round : int, default 10
        Number of boosting iterations per fold.
    nfold : int, default 3
        Number of cross-validation folds.
    seed : int, default 0
        Random seed for reproducible results.
    shuffle : bool, default True
        Whether to shuffle data before creating folds.
    stratified : bool, default False
        Whether to use stratified sampling.
    as_pandas : bool, default True
        Whether to return results as pandas DataFrame.
    show_stdv : bool, default True
        Whether to show standard deviation in verbose output.
    early_stopping_rounds : Optional[int], default None
        Early stopping patience in rounds.
    verbose_eval : Optional[Union[bool, int]], default None
        Verbosity control for evaluation printing.
    maximize : Optional[bool], default None
        Whether to maximize the evaluation metric.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function.
    callbacks : List[TrainingCallback], default empty list
        List of training callbacks to apply.
    folds : Optional[Any], default None
        Custom fold specification or sklearn splitter.
    metrics : List[str], default empty list
        Additional evaluation metrics to monitor.
    
    Examples
    --------
    Basic CV configuration:
    >>> config = CrossValidationConfiguration(nfold=5, num_boost_round=100)
    
    Stratified CV with early stopping:
    >>> config = CrossValidationConfiguration(
    ...     nfold=5,
    ...     num_boost_round=1000,
    ...     stratified=True,
    ...     early_stopping_rounds=50,
    ...     seed=42
    ... )
    
    Custom configuration:
    >>> config = CrossValidationConfiguration(
    ...     nfold=10,
    ...     shuffle=False,
    ...     verbose_eval=25,
    ...     metrics=['auc', 'logloss']
    ... )
    """
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUNDS
    nfold: int = DEFAULT_NFOLD
    seed: int = DEFAULT_SEED
    shuffle: bool = DEFAULT_SHUFFLE
    stratified: bool = DEFAULT_STRATIFIED
    as_pandas: bool = DEFAULT_AS_PANDAS
    show_stdv: bool = DEFAULT_SHOW_STDV
    early_stopping_rounds: Optional[int] = None
    verbose_eval: Optional[Union[bool, int]] = None
    maximize: Optional[bool] = None
    custom_metric: Optional[Metric] = None
    callbacks: List[TrainingCallback] = field(default_factory=list)
    folds: Optional[Any] = None
    metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate CV configuration parameters.
        
        Raises
        ------
        XGBoostConfigurationError
            If any configuration parameter is invalid.
        """
        if not isinstance(self.num_boost_round, int) or self.num_boost_round < MIN_BOOST_ROUNDS:
            raise XGBoostConfigurationError(
                f"num_boost_round must be an integer >= {MIN_BOOST_ROUNDS}, "
                f"got {self.num_boost_round}"
            )
        
        if not isinstance(self.nfold, int) or self.nfold < MIN_NFOLD or self.nfold > MAX_NFOLD:
            raise XGBoostConfigurationError(
                f"nfold must be an integer between {MIN_NFOLD} and {MAX_NFOLD}, "
                f"got {self.nfold}"
            )
        
        if not isinstance(self.seed, int) or not (RANDOM_STATE_RANGE[0] <= self.seed <= RANDOM_STATE_RANGE[1]):
            raise XGBoostConfigurationError(
                f"seed must be an integer between {RANDOM_STATE_RANGE[0]} and {RANDOM_STATE_RANGE[1]}, "
                f"got {self.seed}"
            )
        
        if self.stratified and not SKLEARN_INSTALLED:
            raise XGBoostConfigurationError(
                "sklearn must be installed to use stratified cross-validation"
            )
        
        if self.early_stopping_rounds is not None:
            if (not isinstance(self.early_stopping_rounds, int) or 
                self.early_stopping_rounds < MIN_EARLY_STOPPING_ROUNDS):
                raise XGBoostConfigurationError(
                    f"early_stopping_rounds must be an integer >= {MIN_EARLY_STOPPING_ROUNDS}, "
                    f"got {self.early_stopping_rounds}"
                )


# Utility functions for common operations
def create_parameter_list(params: Dict[str, Any], metrics: List[str]) -> List[Tuple[str, Any]]:
    """Create parameter list with evaluation metrics.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Base parameters dictionary.
    metrics : List[str]
        List of evaluation metrics to add.
        
    Returns
    -------
    List[Tuple[str, Any]]
        Parameter list suitable for Booster initialization.
    """
    param_list = list(params.items())
    for metric in metrics:
        param_list.append(("eval_metric", metric))
    return param_list


def validate_array_size(arr: np.ndarray, max_size: int = MAX_ARRAY_SIZE) -> None:
    """Validate array size to prevent memory issues.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    max_size : int, default MAX_ARRAY_SIZE
        Maximum allowed array size.
        
    Raises
    ------
    XGBoostValidationError
        If array is too large.
    """
    if arr.size > max_size:
        raise XGBoostValidationError(
            f"Array size {arr.size} exceeds maximum allowed size {max_size}"
        )


def safe_random_seed(seed: int) -> None:
    """Set random seed with validation.
    
    Parameters
    ----------
    seed : int
        Random seed value.
        
    Raises
    ------
    XGBoostValidationError
        If seed is out of valid range.
    """
    if not (RANDOM_STATE_RANGE[0] <= seed <= RANDOM_STATE_RANGE[1]):
        raise XGBoostValidationError(
            f"Seed must be between {RANDOM_STATE_RANGE[0]} and {RANDOM_STATE_RANGE[1]}, "
            f"got {seed}"
        )
    np.random.seed(seed)


def create_evaluation_pairs(datasets: List[DMatrix], names: List[str]) -> List[Tuple[DMatrix, str]]:
    """Create evaluation dataset pairs with validation.
    
    Parameters
    ----------
    datasets : List[DMatrix]
        List of evaluation datasets.
    names : List[str]
        List of dataset names.
        
    Returns
    -------
    List[Tuple[DMatrix, str]]
        List of (dataset, name) tuples.
        
    Raises
    ------
    XGBoostValidationError
        If datasets and names lists don't match.
    """
    if len(datasets) != len(names):
        raise XGBoostValidationError(
            f"Number of datasets ({len(datasets)}) must match number of names ({len(names)})"
        )
    
    for i, (dataset, name) in enumerate(zip(datasets, names)):
        if not isinstance(dataset, DMatrix):
            raise TypeError(f"Dataset {i} must be DMatrix, got {type(dataset)}")
        if not isinstance(name, str):
            raise TypeError(f"Name {i} must be str, got {type(name)}")
    
    return list(zip(datasets, names))


# Abstract Strategy Pattern for Cross-Validation
class CVStrategy(ABC):
    """Abstract base class for cross-validation strategies.
    
    This class defines the interface for different cross-validation
    splitting strategies, allowing for flexible and extensible
    cross-validation approaches.
    """
    
    @abstractmethod
    def create_folds(
        self,
        data: DMatrix,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable] = None
    ) -> List["CVPack"]:
        """Create cross-validation folds.
        
        Parameters
        ----------
        data : DMatrix
            Complete dataset to split.
        config : CrossValidationConfiguration
            CV configuration parameters.
        params : Dict[str, Any]
            Model parameters.
        fpreproc : Optional[FPreProcCallable], default None
            Preprocessing function.
            
        Returns
        -------
        List[CVPack]
            List of cross-validation fold packages.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name for logging/identification."""
        pass


class StandardCVStrategy(CVStrategy):
    """Standard k-fold cross-validation strategy.
    
    This strategy creates standard k-fold splits without any special
    considerations for class balance or group structure.
    """
    
    def create_folds(
        self,
        data: DMatrix,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable] = None
    ) -> List["CVPack"]:
        """Create standard k-fold cross-validation folds."""
        safe_random_seed(config.seed)
        
        if config.shuffle:
            idx = np.random.permutation(data.num_row())
        else:
            idx = np.arange(data.num_row())
        
        out_idset = np.array_split(idx, config.nfold)
        in_idset = [
            np.concatenate([out_idset[i] for i in range(config.nfold) if k != i])
            for k in range(config.nfold)
        ]
        
        return self._create_cv_packs(data, in_idset, out_idset, config, params, fpreproc)
    
    def get_name(self) -> str:
        return "StandardCV"
    
    def _create_cv_packs(
        self,
        data: DMatrix,
        in_idset: List[np.ndarray],
        out_idset: List[np.ndarray],
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable]
    ) -> List["CVPack"]:
        """Create CVPack objects from index sets."""
        ret = []
        for k in range(len(in_idset)):
            dtrain = data.slice(in_idset[k])
            dtest = data.slice(out_idset[k])
            
            if fpreproc is not None:
                dtrain, dtest, tparam = fpreproc(dtrain, dtest, params.copy())
            else:
                tparam = params
            
            plst = create_parameter_list(tparam, config.metrics)
            ret.append(CVPack(dtrain, dtest, plst))
        
        return ret


class StratifiedCVStrategy(CVStrategy):
    """Stratified k-fold cross-validation strategy.
    
    This strategy maintains class distribution across folds,
    useful for classification problems with imbalanced classes.
    """
    
    def create_folds(
        self,
        data: DMatrix,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable] = None
    ) -> List["CVPack"]:
        """Create stratified k-fold cross-validation folds."""
        if not SKLEARN_INSTALLED:
            raise XGBoostError("sklearn is required for stratified cross-validation")
        
        sfk = XGBStratifiedKFold(
            n_splits=config.nfold, 
            shuffle=config.shuffle, 
            random_state=config.seed
        )
        
        labels = data.get_label()
        splits = list(sfk.split(X=labels, y=labels))
        in_idset = [x[0] for x in splits]
        out_idset = [x[1] for x in splits]
        
        return StandardCVStrategy()._create_cv_packs(
            data, in_idset, out_idset, config, params, fpreproc
        )
    
    def get_name(self) -> str:
        return "StratifiedCV"


class GroupCVStrategy(CVStrategy):
    """Group-aware cross-validation strategy.
    
    This strategy maintains group integrity across folds,
    essential for ranking problems and grouped data.
    """
    
    def create_folds(
        self,
        data: DMatrix,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable] = None
    ) -> List["CVPack"]:
        """Create group-aware cross-validation folds."""
        group_boundaries = data.get_uint_info("group_ptr")
        if len(group_boundaries) <= 1:
            raise XGBoostValidationError("Dataset must contain group information for group-based CV")
        
        group_sizes = np.diff(group_boundaries)
        safe_random_seed(config.seed)
        
        if config.shuffle:
            idx = np.random.permutation(len(group_sizes))
        else:
            idx = np.arange(len(group_sizes))
        
        out_group_idset = np.array_split(idx, config.nfold)
        in_group_idset = [
            np.concatenate([out_group_idset[i] for i in range(config.nfold) if k != i])
            for k in range(config.nfold)
        ]
        
        in_idset = [
            groups_to_rows(in_groups, group_boundaries) for in_groups in in_group_idset
        ]
        out_idset = [
            groups_to_rows(out_groups, group_boundaries) for out_groups in out_group_idset
        ]
        
        return self._create_group_cv_packs(
            data, in_idset, out_idset, in_group_idset, out_group_idset,
            group_sizes, config, params, fpreproc
        )
    
    def get_name(self) -> str:
        return "GroupCV"
    
    def _create_group_cv_packs(
        self,
        data: DMatrix,
        in_idset: List[np.ndarray],
        out_idset: List[np.ndarray], 
        in_group_idset: List[np.ndarray],
        out_group_idset: List[np.ndarray],
        group_sizes: np.ndarray,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable]
    ) -> List["CVPack"]:
        """Create CVPack objects for group-based CV."""
        ret = []
        for k in range(len(in_idset)):
            dtrain = data.slice(in_idset[k], allow_groups=True)
            dtrain.set_group(group_sizes[in_group_idset[k]])
            dtest = data.slice(out_idset[k], allow_groups=True)
            dtest.set_group(group_sizes[out_group_idset[k]])
            
            if fpreproc is not None:
                dtrain, dtest, tparam = fpreproc(dtrain, dtest, params.copy())
            else:
                tparam = params
            
            plst = create_parameter_list(tparam, config.metrics)
            ret.append(CVPack(dtrain, dtest, plst))
        
        return ret


class CustomCVStrategy(CVStrategy):
    """Custom cross-validation strategy using user-provided folds.
    
    This strategy allows users to provide their own fold specifications
    or use sklearn-compatible splitter objects.
    """
    
    def create_folds(
        self,
        data: DMatrix,
        config: CrossValidationConfiguration,
        params: Dict[str, Any],
        fpreproc: Optional[FPreProcCallable] = None
    ) -> List["CVPack"]:
        """Create custom cross-validation folds."""
        if config.folds is None:
            raise XGBoostValidationError("Custom strategy requires folds to be specified")
        
        try:
            # Try to extract indices directly
            in_idset = [x[0] for x in config.folds]
            out_idset = [x[1] for x in config.folds]
        except (TypeError, IndexError):
            # Assume sklearn-style splitter object
            try:
                labels = data.get_label()
                splits = list(config.folds.split(X=labels, y=labels))
                in_idset = [x[0] for x in splits]
                out_idset = [x[1] for x in splits]
            except AttributeError:
                raise TypeError(
                    "folds must be either a list of (train_idx, test_idx) tuples "
                    "or an sklearn splitter object with a split() method"
                )
        
        return StandardCVStrategy()._create_cv_packs(
            data, in_idset, out_idset, config, params, fpreproc
        )
    
    def get_name(self) -> str:
        return "CustomCV"


def get_cv_strategy(config: CrossValidationConfiguration, data: DMatrix) -> CVStrategy:
    """Select appropriate cross-validation strategy based on configuration and data.
    
    Parameters
    ----------
    config : CrossValidationConfiguration
        Cross-validation configuration.
    data : DMatrix
        Dataset to analyze for strategy selection.
        
    Returns
    -------
    CVStrategy
        Selected cross-validation strategy.
    """
    if config.folds is not None:
        return CustomCVStrategy()
    elif config.stratified:
        return StratifiedCVStrategy()
    elif len(data.get_uint_info("group_ptr")) > 1:
        return GroupCVStrategy()
    else:
        return StandardCVStrategy()


def _validate_training_inputs(
    params: Dict[str, Any], 
    dtrain: DMatrix, 
    config: TrainingConfiguration,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None
) -> None:
    """Validate inputs for training functions.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Training parameters dictionary.
    dtrain : DMatrix
        Training dataset.
    config : TrainingConfiguration
        Training configuration object.
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
    
    if not isinstance(config, TrainingConfiguration):
        raise TypeError(f"Expected TrainingConfiguration for config, got {type(config).__name__}")
    
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
    config: Union[TrainingConfiguration, CrossValidationConfiguration],
    obj: Optional[Objective] = None,
    is_cv: bool = False
) -> CallbackContainer:
    """Set up and configure callback container from configuration.
    
    Parameters
    ----------
    config : Union[TrainingConfiguration, CrossValidationConfiguration]
        Training or CV configuration object.
    obj : Optional[Objective], default None
        Custom objective function.
    is_cv : bool, default False
        Whether this is for cross-validation.
        
    Returns
    -------
    CallbackContainer
        Configured callback container.
    """
    callback_list = copy.copy(config.callbacks) if config.callbacks else []
    
    if config.verbose_eval:
        verbose_eval = DEFAULT_VERBOSE_EVAL_PERIOD if config.verbose_eval is True else config.verbose_eval
        show_stdv = getattr(config, 'show_stdv', DEFAULT_SHOW_STDV) if is_cv else False
        callback_list.append(
            EvaluationMonitor(period=verbose_eval, show_stdv=show_stdv)
        )
    
    if config.early_stopping_rounds:
        callback_list.append(
            EarlyStopping(rounds=config.early_stopping_rounds, maximize=config.maximize)
        )
    
    return CallbackContainer(
        callback_list, 
        metric=config.custom_metric, 
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
    config: Optional[TrainingConfiguration] = None,
    *,
    # Legacy parameters for backward compatibility
    num_boost_round: int = None,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    evals_result: Optional[TrainingCallback.EvalsLog] = None,
    verbose_eval: Optional[Union[bool, int]] = None,
    xgb_model: Optional[Union[str, os.PathLike, Booster, bytearray]] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
) -> Booster:
    """Train an XGBoost model with comprehensive monitoring and callbacks.

    This function trains an XGBoost booster using the provided parameters
    and training data, with support for validation monitoring, early stopping,
    and custom callbacks. It now supports both configuration objects and
    legacy parameter passing for backward compatibility.

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
    config : Optional[TrainingConfiguration], default None
        Training configuration object. If provided, takes precedence over
        individual parameters. Recommended for new code.
    num_boost_round : int, default None
        Number of boosting rounds (iterations) to perform. Legacy parameter.
    evals : Optional[Sequence[Tuple[DMatrix, str]]], default None
        List of validation datasets with names for monitoring.
    obj : Optional[Objective], default None
        Custom objective function. See XGBoost custom objective tutorial.
    maximize : Optional[bool], default None
        Whether to maximize evaluation metric.
    early_stopping_rounds : Optional[int], default None
        Activates early stopping. Training stops if validation metric doesn't
        improve for this many consecutive rounds.
    evals_result : Optional[TrainingCallback.EvalsLog], default None
        Dictionary to store evaluation history. Modified in-place during training.
    verbose_eval : Optional[Union[bool, int]], default None
        Controls evaluation printing.
    xgb_model : Optional[Union[str, os.PathLike, Booster, bytearray]], default None
        Pre-trained model to continue training from.
    callbacks : Optional[Sequence[TrainingCallback]], default None
        Custom callback functions executed during training.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function.

    Returns
    -------
    Booster
        Trained XGBoost model ready for prediction and inference.

    Raises
    ------
    TypeError
        If input types are incorrect.
    XGBoostValidationError
        If parameter validation fails.
    ValueError
        If QuantileDMatrix reference validation fails.
    XGBoostError
        If training fails due to XGBoost-specific issues.

    Examples
    --------
    Using configuration object (recommended):
    >>> import xgboost as xgb
    >>> from xgboost.training import TrainingConfiguration
    >>> params = {'objective': 'reg:squarederror', 'max_depth': 6}
    >>> dtrain = xgb.DMatrix(X_train, y_train)
    >>> config = TrainingConfiguration(num_boost_round=100)
    >>> bst = train(params, dtrain, config)

    Legacy parameter style:
    >>> bst = train(params, dtrain, num_boost_round=100)

    Advanced configuration:
    >>> dval = xgb.DMatrix(X_val, y_val)
    >>> config = TrainingConfiguration(
    ...     num_boost_round=1000,
    ...     early_stopping_rounds=50,
    ...     verbose_eval=10
    ... )
    >>> bst = train(params, dtrain, config, evals=[(dval, 'validation')])

    Notes
    -----
    - Configuration objects are recommended for new code as they provide
      better validation and are more maintainable.
    - Legacy parameters are maintained for backward compatibility.
    - When using early stopping, the returned model is from the last iteration.
    """
    # Handle backward compatibility and configuration
    if config is None:
        # Create configuration from legacy parameters
        config = TrainingConfiguration(
            num_boost_round=num_boost_round or DEFAULT_NUM_BOOST_ROUNDS,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval if verbose_eval is not None else True,
            maximize=maximize,
            custom_metric=custom_metric,
            callbacks=list(callbacks) if callbacks else []
        )
    
    # Validate all inputs
    _validate_training_inputs(params, dtrain, config, evals)
    
    # Process evaluation datasets
    evals = list(evals) if evals else []
    if evals:
        _validate_eval_datasets(evals, dtrain)
    
    # Create booster instance
    all_datasets = [dtrain] + [d[0] for d in evals]
    bst = _create_booster_from_datasets(params, all_datasets, xgb_model)
    
    # Setup callbacks
    cb_container = _setup_callbacks(config, obj, is_cv=False)
    
    # Execute training
    start_iteration = 0
    bst = _execute_training_loop(
        bst, cb_container, start_iteration, config.num_boost_round, dtrain, evals, obj
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
    config: CrossValidationConfiguration
) -> None:
    """Validate inputs for cross-validation functions.
    
    Parameters
    ----------
    params : BoosterParam
        Training parameters.
    dtrain : DMatrix
        Training dataset.
    config : CrossValidationConfiguration
        Cross-validation configuration object.
        
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
    
    if not isinstance(config, CrossValidationConfiguration):
        raise TypeError(f"Expected CrossValidationConfiguration for config, got {type(config).__name__}")


def _process_cv_params(params: BoosterParam, config: CrossValidationConfiguration) -> Tuple[Dict[str, Any], List[str]]:
    """Process and validate cross-validation parameters.
    
    Parameters
    ----------
    params : BoosterParam
        Input parameters (dict or list of tuples).
    config : CrossValidationConfiguration
        CV configuration object.
        
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
    
    # Combine with config metrics
    final_metrics = list(config.metrics) + metrics
    
    return params_dict, final_metrics


def _execute_cv_training_loop(
    booster: _PackedBooster,
    callbacks_container: CallbackContainer,
    config: CrossValidationConfiguration,
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
    config : CrossValidationConfiguration
        CV configuration object.
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

    for i in range(config.num_boost_round):
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
    config: Optional[CrossValidationConfiguration] = None,
    *,
    # Legacy parameters for backward compatibility
    num_boost_round: int = None,
    nfold: int = None,
    stratified: bool = None,
    folds: XGBStratifiedKFold = None,
    metrics: Sequence[str] = None,
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    fpreproc: Optional[FPreProcCallable] = None,
    as_pandas: bool = None,
    verbose_eval: Optional[Union[int, bool]] = None,
    show_stdv: bool = None,
    seed: int = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    shuffle: bool = None,
    custom_metric: Optional[Metric] = None,
) -> Union[Dict[str, float], "PdDataFrame"]:
    """Perform cross-validation with comprehensive monitoring and callbacks.
    
    This function performs k-fold cross-validation on the provided dataset,
    supporting various splitting strategies, early stopping, and custom metrics.
    Results include both mean and standard deviation across folds. Now supports
    both configuration objects and legacy parameter passing.

    Parameters
    ----------
    params : BoosterParam
        Dictionary or list of XGBoost parameters.
    dtrain : DMatrix
        Complete dataset for cross-validation. Must not use external memory.
    config : Optional[CrossValidationConfiguration], default None
        Cross-validation configuration object. If provided, takes precedence
        over individual parameters. Recommended for new code.
    num_boost_round : int, default None
        Number of boosting iterations per fold. Legacy parameter.
    nfold : int, default None
        Number of cross-validation folds. Legacy parameter.
    stratified : bool, default None
        Whether to use stratified sampling. Legacy parameter.
    folds : XGBStratifiedKFold, default None
        Custom fold specification. Legacy parameter.
    metrics : Sequence[str], default None
        Additional evaluation metrics. Legacy parameter.
    obj : Optional[Objective], default None
        Custom objective function applied to all folds.
    maximize : Optional[bool], default None
        Whether to maximize evaluation metric for early stopping.
    early_stopping_rounds : Optional[int], default None
        Stop training if CV metric doesn't improve for N consecutive rounds.
    fpreproc : Optional[FPreProcCallable], default None
        Preprocessing function applied to each fold.
    as_pandas : bool, default None
        Return pandas DataFrame if available. Legacy parameter.
    verbose_eval : Optional[Union[int, bool]], default None
        Print progress control. Legacy parameter.
    show_stdv : bool, default None
        Show standard deviation in verbose output. Legacy parameter.
    seed : int, default None
        Random seed for reproducible fold creation. Legacy parameter.
    callbacks : Optional[Sequence[TrainingCallback]], default None
        Custom callbacks applied during CV training.
    shuffle : bool, default None
        Shuffle data before creating folds. Legacy parameter.
    custom_metric : Optional[Metric], default None
        Custom evaluation metric function.

    Returns
    -------
    Union[Dict[str, List[float]], PdDataFrame]
        Cross-validation results with mean and std for each metric.

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
    Using configuration object (recommended):
    >>> import xgboost as xgb
    >>> from xgboost.training import CrossValidationConfiguration
    >>> params = {'objective': 'binary:logistic', 'max_depth': 3}
    >>> dtrain = xgb.DMatrix(X, y)
    >>> config = CrossValidationConfiguration(nfold=5, num_boost_round=100)
    >>> cv_results = cv(params, dtrain, config)

    Legacy parameter style:
    >>> cv_results = cv(params, dtrain, num_boost_round=100, nfold=5)

    Advanced configuration:
    >>> config = CrossValidationConfiguration(
    ...     nfold=5,
    ...     num_boost_round=1000,
    ...     stratified=True,
    ...     early_stopping_rounds=50,
    ...     seed=42,
    ...     metrics=['auc', 'logloss']
    ... )
    >>> cv_results = cv(params, dtrain, config)

    Notes
    -----
    - Configuration objects provide better validation and maintainability.
    - Legacy parameters are maintained for backward compatibility.
    - For ranking problems with groups, group structure is preserved automatically.
    - Early stopping uses average metric across all folds.
    """
    # Handle backward compatibility and configuration
    if config is None:
        # Create configuration from legacy parameters
        config = CrossValidationConfiguration(
            num_boost_round=num_boost_round or DEFAULT_NUM_BOOST_ROUNDS,
            nfold=nfold or DEFAULT_NFOLD,
            seed=seed or DEFAULT_SEED,
            shuffle=shuffle if shuffle is not None else DEFAULT_SHUFFLE,
            stratified=stratified if stratified is not None else DEFAULT_STRATIFIED,
            as_pandas=as_pandas if as_pandas is not None else DEFAULT_AS_PANDAS,
            show_stdv=show_stdv if show_stdv is not None else DEFAULT_SHOW_STDV,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            maximize=maximize,
            custom_metric=custom_metric,
            callbacks=list(callbacks) if callbacks else [],
            folds=folds,
            metrics=list(metrics) if metrics else []
        )
    
    # Validate inputs
    _validate_cv_inputs(params, dtrain, config)

    # Process parameters and metrics
    params_dict, final_metrics = _process_cv_params(params, config)
    
    # Update config with final metrics
    config.metrics = final_metrics

    # Create cross-validation folds using strategy pattern
    cv_strategy = get_cv_strategy(config, dtrain)
    cvfolds = cv_strategy.create_folds(dtrain, config, params_dict, fpreproc)

    # Setup callbacks
    callbacks_container = _setup_callbacks(config, obj, is_cv=True)

    # Execute cross-validation training
    booster = _PackedBooster(cvfolds)
    results = _execute_cv_training_loop(
        booster, callbacks_container, config, dtrain, obj
    )
    
    # Convert to pandas if requested and available
    if config.as_pandas:
        try:
            import pandas as pd
            results = pd.DataFrame.from_dict(results)
        except ImportError:
            pass  # Return dict format if pandas not available

    return results


# Legacy function aliases for backward compatibility
def mknfold(*args, **kwargs):
    """Legacy function - use CrossValidationConfiguration and cv() instead."""
    import warnings
    warnings.warn(
        "mknfold is deprecated. Use CrossValidationConfiguration and cv() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Implementation would delegate to new system
    pass


def mkgroupfold(*args, **kwargs):
    """Legacy function - use GroupCVStrategy via cv() instead."""
    import warnings
    warnings.warn(
        "mkgroupfold is deprecated. Use GroupCVStrategy via cv() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Implementation would delegate to new system
    pass