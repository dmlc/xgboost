# type: ignore
"""Xgboost pyspark integration submodule for helper functions."""
import inspect
import logging
import sys
from threading import Thread

import pyspark
from pyspark.sql.session import SparkSession
from xgboost.tracker import RabitTracker

from xgboost import rabit


def get_class_name(cls):
    """
    Return the class name.
    """
    return f"{cls.__module__}.{cls.__name__}"


def _get_default_params_from_func(func, unsupported_set):
    """
    Returns a dictionary of parameters and their default value of function fn.
    Only the parameters with a default value will be included.
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


class RabitContext:
    """
    A context controlling rabit initialization and finalization.
    This isn't specificially necessary (note Part 3), but it is more understandable coding-wise.
    """

    def __init__(self, args, context):
        self.args = args
        self.args.append(("DMLC_TASK_ID=" + str(context.partitionId())).encode())

    def __enter__(self):
        rabit.init(self.args)

    def __exit__(self, *args):
        rabit.finalize()


def _start_tracker(context, n_workers):
    """
    Start Rabit tracker with n_workers
    """
    env = {"DMLC_NUM_WORKER": n_workers}
    host = _get_host_ip(context)
    rabit_context = RabitTracker(host_ip=host, n_workers=n_workers)
    env.update(rabit_context.worker_envs())
    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _get_rabit_args(context, n_workers):
    """
    Get rabit context arguments to send to each worker.
    """
    # pylint: disable=consider-using-f-string
    env = _start_tracker(context, n_workers)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]
    return rabit_args


def _get_host_ip(context):
    """
    Gets the hostIP for Spark. This essentially gets the IP of the first worker.
    """
    task_ip_list = [info.address.split(":")[0] for info in context.getTaskInfos()]
    return task_ip_list[0]


def _get_args_from_message_list(messages):
    """
    A function to send/recieve messages in barrier context mode
    """
    output = ""
    for message in messages:
        if message != "":
            output = message
            break
    return [elem.split("'")[1].encode() for elem in output.strip("][").split(", ")]


def _get_spark_session():
    """Get or create spark session. Note: This function can only be invoked from driver side."""
    if pyspark.TaskContext.get() is not None:
        # This is a safety check.
        raise RuntimeError(
            "_get_spark_session should not be invoked from executor side."
        )
    return SparkSession.builder.getOrCreate()


def get_logger(name, level="INFO"):
    """Gets a logger by name, or creates and configures it for the first time."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    return logger


def _get_max_num_concurrent_tasks(spark_context):
    """Gets the current max number of concurrent tasks."""
    # pylint: disable=protected-access
    # spark 3.1 and above has a different API for fetching max concurrent tasks
    if spark_context._jsc.sc().version() >= "3.1":
        return spark_context._jsc.sc().maxNumConcurrentTasks(
            spark_context._jsc.sc().resourceProfileManager().resourceProfileFromId(0)
        )
    return spark_context._jsc.sc().maxNumConcurrentTasks()


def _is_local(spark_context) -> bool:
    """Whether it is Spark local mode"""
    # pylint: disable=protected-access
    return spark_context._jsc.sc().isLocal()


def _get_gpu_id(task_context) -> int:
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
