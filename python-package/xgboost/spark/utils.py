"""Xgboost pyspark integration submodule for helper functions."""

# pylint: disable=fixme

import inspect
import logging
import os
import sys
import uuid
from threading import Thread
from typing import Any, Callable, Dict, Optional, Set, Type, Union

import pyspark
from pyspark import BarrierTaskContext, SparkConf, SparkContext, SparkFiles, TaskContext
from pyspark.sql.session import SparkSession

from xgboost import Booster, XGBModel
from xgboost.collective import CommunicatorContext as CCtx
from xgboost.tracker import RabitTracker


def get_class_name(cls: Type) -> str:
    """Return the class name."""
    return f"{cls.__module__}.{cls.__name__}"


def _get_default_params_from_func(
    func: Callable, unsupported_set: Set[str]
) -> Dict[str, Any]:
    """Returns a dictionary of parameters and their default value of function fn.  Only
    the parameters with a default value will be included.

    """
    sig = inspect.signature(func)
    filtered_params_dict = {}
    for parameter in sig.parameters.values():
        # Remove parameters without a default value and those in the unsupported_set
        if (
            parameter.default is not parameter.empty
            and parameter.name not in unsupported_set
        ):
            filtered_params_dict[parameter.name] = parameter.default
    return filtered_params_dict


class CommunicatorContext(CCtx):
    """Context with PySpark specific task ID."""

    def __init__(self, context: BarrierTaskContext, **args: Any) -> None:
        args["dmlc_task_id"] = str(context.partitionId())
        super().__init__(**args)


def _start_tracker(context: BarrierTaskContext, n_workers: int) -> Dict[str, Any]:
    """Start Rabit tracker with n_workers"""
    args: Dict[str, Any] = {"n_workers": n_workers}
    host = _get_host_ip(context)
    tracker = RabitTracker(n_workers=n_workers, host_ip=host, sortby="task")
    tracker.start()
    thread = Thread(target=tracker.wait_for)
    thread.daemon = True
    thread.start()
    args.update(tracker.worker_args())
    return args


def _get_rabit_args(context: BarrierTaskContext, n_workers: int) -> Dict[str, Any]:
    """Get rabit context arguments to send to each worker."""
    env = _start_tracker(context, n_workers)
    return env


def _get_host_ip(context: BarrierTaskContext) -> str:
    """Gets the hostIP for Spark. This essentially gets the IP of the first worker."""
    task_ip_list = [info.address.split(":")[0] for info in context.getTaskInfos()]
    return task_ip_list[0]


def _get_spark_session() -> SparkSession:
    """Get or create spark session. Note: This function can only be invoked from driver
    side.

    """
    if pyspark.TaskContext.get() is not None:
        # This is a safety check.
        raise RuntimeError(
            "_get_spark_session should not be invoked from executor side."
        )
    return SparkSession.builder.getOrCreate()


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Gets a logger by name, or creates and configures it for the first time."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    else:
        # Default to info if not set.
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(funcName)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger_level(name: str) -> Optional[int]:
    """Get the logger level for the given log name"""
    logger = logging.getLogger(name)
    return None if logger.level == logging.NOTSET else logger.level


def _get_max_num_concurrent_tasks(spark_context: SparkContext) -> int:
    """Gets the current max number of concurrent tasks."""
    # pylint: disable=protected-access
    # spark 3.1 and above has a different API for fetching max concurrent tasks
    if spark_context._jsc.sc().version() >= "3.1":
        return spark_context._jsc.sc().maxNumConcurrentTasks(
            spark_context._jsc.sc().resourceProfileManager().resourceProfileFromId(0)
        )
    return spark_context._jsc.sc().maxNumConcurrentTasks()


def _is_local(spark_context: SparkContext) -> bool:
    """Whether it is Spark local mode"""
    # pylint: disable=protected-access
    return spark_context._jsc.sc().isLocal()


def _is_standalone_or_localcluster(conf: SparkConf) -> bool:
    master = conf.get("spark.master")
    return master is not None and (
        master.startswith("spark://") or master.startswith("local-cluster")
    )


def _get_gpu_id(task_context: TaskContext) -> int:
    """Get the gpu id from the task resources"""
    if task_context is None:
        # This is a safety check.
        raise RuntimeError("_get_gpu_id should not be invoked from driver side.")
    resources = task_context.resources()
    if "gpu" not in resources:
        raise RuntimeError(
            "Couldn't get the gpu id, Please check the GPU resource configuration"
        )
    # return the first gpu id.
    return int(resources["gpu"].addresses[0].strip())


def _get_or_create_tmp_dir() -> str:
    root_dir = SparkFiles.getRootDirectory()
    xgb_tmp_dir = os.path.join(root_dir, "xgboost-tmp")
    if not os.path.exists(xgb_tmp_dir):
        os.makedirs(xgb_tmp_dir)
    return xgb_tmp_dir


def deserialize_xgb_model(
    model: str, xgb_model_creator: Callable[[], XGBModel]
) -> XGBModel:
    """
    Deserialize an xgboost.XGBModel instance from the input model.
    """
    xgb_model = xgb_model_creator()
    xgb_model.load_model(bytearray(model.encode("utf-8")))
    return xgb_model


def serialize_booster(booster: Booster) -> str:
    """
    Serialize the input booster to a string.

    Parameters
    ----------
    booster:
        an xgboost.core.Booster instance
    """
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f"{uuid.uuid4()}.json")
    booster.save_model(tmp_file_name)
    with open(tmp_file_name, encoding="utf-8") as f:
        ser_model_string = f.read()
    return ser_model_string


def deserialize_booster(model: str) -> Booster:
    """
    Deserialize an xgboost.core.Booster from the input ser_model_string.
    """
    booster = Booster()
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f"{uuid.uuid4()}.json")
    with open(tmp_file_name, "w", encoding="utf-8") as f:
        f.write(model)
    booster.load_model(tmp_file_name)
    return booster


def use_cuda(device: Optional[str]) -> bool:
    """Whether xgboost is using CUDA workers."""
    return device in ("cuda", "gpu")
