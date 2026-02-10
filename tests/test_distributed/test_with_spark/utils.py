import logging
import os
import shutil
import tempfile
import unittest

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
        if os.environ.get("XGBOOST_PYSPARK_SHARED_SESSION") == "1":
            cls.session = None
            cls.sc = None
            return
        cls.session.stop()
        cls.session = None
        cls.sc.stop()
        cls.sc = None


class SparkLocalClusterTestCase(TestSparkContext, TestTempDir, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup_env(
            {
                "spark.master": "local-cluster[2, 1, 1024]",
                "spark.python.worker.reuse": "true",
                "spark.driver.host": "127.0.0.1",
                "spark.task.maxFailures": "1",
                "spark.sql.shuffle.partitions": "4",
                "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
                "spark.sql.pyspark.jvmStacktrace.enabled": "true",
                "spark.cores.max": "2",
                "spark.task.cpus": "1",
                "spark.executor.cores": "1",
                "spark.ui.enabled": "false",
            }
        )
        cls.make_tempdir()
        # We run a dummy job so that we block until the workers have connected to the master
        num_slots = cls.sc.defaultParallelism
        cls.sc.parallelize(range(num_slots), num_slots).barrier().mapPartitions(
            lambda _: []
        ).collect()

    @classmethod
    def tearDownClass(cls):
        cls.remove_tempdir()
        cls.tear_down_env()
