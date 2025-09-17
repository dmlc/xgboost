import contextlib
import logging
import shutil
import sys
import tempfile
import unittest
from io import StringIO

import pytest

from xgboost import testing as tm

pytestmark = [pytest.mark.skipif(**tm.no_spark())]

from pyspark.sql import SparkSession

from xgboost.spark.utils import _get_default_params_from_func


class UtilsTest(unittest.TestCase):
    def test_get_default_params(self):
        class Foo:
            def func1(self, x, y, key1=None, key2="val2", key3=0, key4=None):
                pass

        unsupported_params = {"key2", "key4"}
        expected_default_params = {
            "key1": None,
            "key3": 0,
        }
        actual_default_params = _get_default_params_from_func(
            Foo.func1, unsupported_params
        )
        self.assertEqual(
            len(expected_default_params.keys()), len(actual_default_params.keys())
        )
        for k, v in actual_default_params.items():
            self.assertEqual(expected_default_params[k], v)


@contextlib.contextmanager
def patch_stdout():
    """patch stdout and give an output"""
    sys_stdout = sys.stdout
    io_out = StringIO()
    sys.stdout = io_out
    try:
        yield io_out
    finally:
        sys.stdout = sys_stdout


@contextlib.contextmanager
def patch_logger(name):
    """patch logger and give an output"""
    io_out = StringIO()
    log = logging.getLogger(name)
    handler = logging.StreamHandler(io_out)
    log.addHandler(handler)
    try:
        yield io_out
    finally:
        log.removeHandler(handler)


class TestTempDir(object):
    @classmethod
    def make_tempdir(cls):
        """
        :param dir: Root directory in which to create the temp directory
        """
        cls.tempdir = tempfile.mkdtemp(prefix="sparkdl_tests")

    @classmethod
    def remove_tempdir(cls):
        shutil.rmtree(cls.tempdir)


class TestSparkContext(object):
    @classmethod
    def setup_env(cls, spark_config):
        builder = SparkSession.builder.appName("xgboost spark python API Tests")
        for k, v in spark_config.items():
            builder.config(k, v)
        spark = builder.getOrCreate()
        logging.getLogger("pyspark").setLevel(logging.INFO)

        cls.sc = spark.sparkContext
        cls.session = spark

    @classmethod
    def tear_down_env(cls):
        cls.session.stop()
        cls.session = None
        cls.sc.stop()
        cls.sc = None


class SparkTestCase(TestSparkContext, TestTempDir, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup_env(
            {
                "spark.master": "local[4]",
                "spark.python.worker.reuse": "false",
                "spark.driver.host": "127.0.0.1",
                "spark.task.maxFailures": "1",
                "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
                "spark.sql.pyspark.jvmStacktrace.enabled": "true",
            }
        )
        cls.make_tempdir()

    @classmethod
    def tearDownClass(cls):
        cls.remove_tempdir()
        cls.tear_down_env()


class SparkLocalClusterTestCase(TestSparkContext, TestTempDir, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup_env(
            {
                "spark.master": "local-cluster[2, 2, 1024]",
                "spark.python.worker.reuse": "false",
                "spark.driver.host": "127.0.0.1",
                "spark.task.maxFailures": "1",
                "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
                "spark.sql.pyspark.jvmStacktrace.enabled": "true",
                "spark.cores.max": "4",
                "spark.task.cpus": "1",
                "spark.executor.cores": "2",
            }
        )
        cls.make_tempdir()
        # We run a dummy job so that we block until the workers have connected to the master
        cls.sc.parallelize(range(4), 4).barrier().mapPartitions(lambda _: []).collect()

    @classmethod
    def tearDownClass(cls):
        cls.remove_tempdir()
        cls.tear_down_env()
