# XGBoost4j Pyspark API Integration Tests

This integration tests framework refers to [Nvidia/spark-rapids/integration_tests](https://github.com/NVIDIA/spark-rapids/tree/branch-22.04/integration_tests).

## Setting Up the Environment

The tests are based off of `pyspark` and `pytest` running on Python 3. There really are
only a small number of Python dependencies that you need to install for the tests. The
dependencies also only need to be on the driver.  You can install them on all nodes
in the cluster but it is not required.

- install python dependencies

``` bash
pip install pytest numpy scipy
```

- install xgboost python package

XGBoost4j pyspark APIs are in xgboost python package, so we need to install it first

``` bash
cd xgboost/python-packages
python setup.py install
```

- compile xgboost jvm packages 

``` bash
cd xgboost/jvm-packages
mvn -Dmaven.test.skip=true -DskipTests clean package
```

- run integration tests

```bash
cd xgboost/jvm-packages/integration-tests
./run_pyspark_from_build.sh
```
