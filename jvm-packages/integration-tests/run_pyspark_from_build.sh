#!/bin/bash
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH"

if [[ $( echo ${SKIP_TESTS} | tr [:upper:] [:lower:] ) == "true" ]];
then
    echo "PYTHON INTEGRATION TESTS SKIPPED..."
elif [[ -z "$SPARK_HOME" ]];
then
    >&2 echo "SPARK_HOME IS NOT SET CANNOT RUN PYTHON INTEGRATION TESTS..."
else
    echo "WILL RUN TESTS WITH SPARK_HOME: ${SPARK_HOME}"

    # support alternate local jars NOT building from the source code
    if [ -d "$LOCAL_JAR_PATH" ]; then
        XGBOOST_4J_JAR=$(echo "$LOCAL_JAR_PATH"/xgboost4j_2.12-*.jar)
        XGBOOST_4J_SPARK_JAR=$(echo "$LOCAL_JAR_PATH"/xgboost4j-spark_2.12-*.jar)
    else
        XGBOOST_4J_JAR=$(echo "$SCRIPTPATH"/../xgboost4j/target/xgboost4j_2.12-*.jar)
        XGBOOST_4J_SPARK_JAR=$(echo "$SCRIPTPATH"/../xgboost4j-spark/target/xgboost4j-spark_2.12-*.jar)
    fi
    if [ ! -e $XGBOOST_4J_JAR ]; then
        echo "$XGBOOST_4J_JAR does not exist"
        exit 2
    fi
    if [ ! -e $XGBOOST_4J_SPARK_JAR ]; then
        echo "$XGBOOST_4J_SPARK_JAR does not exist"
        exit 2
    fi
    ALL_JARS="$XGBOOST_4J_JAR,$XGBOOST_4J_SPARK_JAR"
    echo "AND XGBoost JARS: $ALL_JARS"

    if [[ "${TEST}" != "" ]];
    then
        TEST_ARGS="-k $TEST"
    fi
    if [[ "${TEST_TAGS}" != "" ]];
    then
        TEST_TAGS="-m $TEST_TAGS"
    fi

    TEST_TYPE_PARAM=""
    if [[ "${TEST_TYPE}" != "" ]];
    then
        TEST_TYPE_PARAM="--test_type $TEST_TYPE"
    fi

    RUN_DIR=${RUN_DIR-"$SCRIPTPATH"/target/run_dir}
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"

    TEST_COMMON_OPTS=(-v
      -rfExXs
      "$TEST_TAGS"
      --color=yes
      --platform='gpu'
      $TEST_TYPE_PARAM
      "$TEST_ARGS"
      $RUN_TEST_PARAMS
      --junitxml=TEST-pytest-`date +%s%N`.xml
      "$@")

    NUM_LOCAL_EXECS=${NUM_LOCAL_EXECS:-0}
    MB_PER_EXEC=${MB_PER_EXEC:-1024}
    CORES_PER_EXEC=${CORES_PER_EXEC:-1}

    SPARK_TASK_MAXFAILURES=1

    export PYSP_TEST_spark_driver_extraClassPath="${ALL_JARS// /:}"
    export PYSP_TEST_spark_executor_extraClassPath="${ALL_JARS// /:}"
    export PYSP_TEST_spark_driver_extraJavaOptions="-ea -Duser.timezone=UTC $COVERAGE_SUBMIT_FLAGS"
    export PYSP_TEST_spark_executor_extraJavaOptions='-ea -Duser.timezone=UTC'
    export PYSP_TEST_spark_ui_showConsoleProgress='false'
    export PYSP_TEST_spark_sql_session_timeZone='UTC'
    # prevent cluster shape to change
    export PYSP_TEST_spark_dynamicAllocation_enabled='false'

    # Set spark.task.maxFailures for most schedulers.
    #
    # Local (non-cluster) mode is the exception and does not work with `spark.task.maxFailures`.
    # It requires two arguments to the master specification "local[N, K]" where
    # N is the number of threads, and K is the maxFailures (otherwise this is hardcoded to 1,
    # see https://issues.apache.org/jira/browse/SPARK-2083).
    export PYSP_TEST_spark_task_maxFailures="1"

    if ((NUM_LOCAL_EXECS > 0)); then
      export PYSP_TEST_spark_master="local-cluster[$NUM_LOCAL_EXECS,$CORES_PER_EXEC,$MB_PER_EXEC]"
    else
      # If a master is not specified, use "local[*, $SPARK_TASK_MAXFAILURES]"
      if [ -z "${PYSP_TEST_spark_master}" ] && [[ "$SPARK_SUBMIT_FLAGS" != *"--master"* ]]; then
        export PYSP_TEST_spark_master="local[*,$SPARK_TASK_MAXFAILURES]"
      fi
    fi

    LOCAL_ROOTDIR=${LOCAL_ROOTDIR:-"$SCRIPTPATH"} 
    RUN_TESTS_COMMAND=("$SCRIPTPATH"/runtests.py
      --rootdir
      "$LOCAL_ROOTDIR"
      "$LOCAL_ROOTDIR"/python)

    exec "$SPARK_HOME"/bin/spark-submit --jars "${ALL_JARS// /,}" \
      --driver-java-options "$PYSP_TEST_spark_driver_extraJavaOptions" \
      $SPARK_SUBMIT_FLAGS "${RUN_TESTS_COMMAND[@]}" "${TEST_COMMON_OPTS[@]}"
fi
