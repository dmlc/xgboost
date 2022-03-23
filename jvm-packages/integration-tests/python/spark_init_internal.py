#
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

try:
    import pyspark
except ImportError as error:
    import findspark
    findspark.init()
    import pyspark

_DRIVER_ENV = 'PYSP_TEST_spark_driver_extraJavaOptions'

def _spark__init():
    # Force the RapidsPlugin to be enabled, so it blows up if the classpath is not set properly
    # DO NOT SET ANY OTHER CONFIGS HERE!!!
    # due to bugs in pyspark/pytest it looks like any configs set here
    # can be reset in the middle of a test if specific operations are done (some types of cast etc)
    _sb = pyspark.sql.SparkSession.builder

    for key, value in os.environ.items():
        if key.startswith('PYSP_TEST_') and key != _DRIVER_ENV:
            _sb.config(key[10:].replace('_', '.'), value)

    driver_opts = os.environ.get(_DRIVER_ENV, "")

    _sb.config('spark.driver.extraJavaOptions', driver_opts)
    _handle_event_log_dir(_sb, 'gw0')

    _s = _sb.appName('xgboost4j pyspark integration tests').getOrCreate()
    # TODO catch the ClassNotFound error that happens if the classpath is not set up properly and
    # make it a better error message
    _s.sparkContext.setLogLevel("WARN")
    return _s


def _handle_event_log_dir(sb, wid):
    if os.environ.get('SPARK_EVENTLOG_ENABLED', str(True)).lower() in [
        str(False).lower(), 'off', '0'
    ]:
        print('Automatic configuration for spark event log disabled')
        return

    spark_conf = pyspark.SparkConf()
    master_url = os.environ.get('PYSP_TEST_spark_master',
                                spark_conf.get("spark.master", 'local'))
    event_log_config = os.environ.get('PYSP_TEST_spark_eventLog_enabled',
                                      spark_conf.get('spark.eventLog.enabled', str(False).lower()))
    event_log_codec = os.environ.get('PYSP_TEST_spark_eventLog_compression_codec', 'zstd')

    if not master_url.startswith('local') or event_log_config != str(False).lower():
        print("SPARK_EVENTLOG_ENABLED is ignored for non-local Spark master and when "
              "it's pre-configured by the user")
        return
    d = "./eventlog_{}".format(wid)
    if not os.path.exists(d):
        os.makedirs(d)

    print('Spark event logs will appear under {}. Set the environmnet variable '
          'SPARK_EVENTLOG_ENABLED=false if you want to disable it'.format(d))

    sb\
        .config('spark.eventLog.dir', "file://{}".format(os.path.abspath(d))) \
        .config('spark.eventLog.compress', True) \
        .config('spark.eventLog.enabled', True) \
        .config('spark.eventLog.compression.codec', event_log_codec)


_spark = _spark__init()


def get_spark():
    """
    Get the current SparkSession.
    """
    return _spark


def spark_version():
    return _spark.version
