## How to run the benchmarks:
1. Obtain python package of XGBoost. There are a few options:
    - Build XGBoost from sources manually:
        ```sh
        git clone --recursive https://github.com/dmlc/xgboost
        cd xgboost
        make -j8
        cd python-package
        python setup.py install
        cd ..
        ```
    - Or download the latest available version from pip:
        ```sh
        pip install xgboost
        ```
    - More details are available [here](https://xgboost.readthedocs.io/en/latest/build.html)

2. Resolve dependencies on other python packages. For now it has dependencies on further packages: requests, scikit-learn, pandas, numpy. You can easily download them through pip:
    ```sh
        pip install requests scikit-learn pandas
    ```
3. Run benchmarks with specified parameters:
    ```sh
    cd tests/benchmark/hist_method
    python xgboost_bench.py  --dataset < dataset > \
                             --hw < platform >     \
                             --n_iter < n_iter >   \
                             --n_runs < n_runs >   \
                             --log < enable_log >
    ```

The benchmark downloads required datasets from the Internet automatically, you don't need to worry about it.

## Available parameters:
* **dataset**    - dataset to use in benchmark. Possible values: *"higgs1m", "airline-ohe", "msrank-10k"* [Required].
* **platform**   - specify platform for computation. Possible values: *cpu, gpu*. [Default=cpu].
* **n_iter**     - amount of boosting iterations. Possible values: *integer > 0*. [Default=1000].
* **n_runs**     - number of training and prediction measurements to obtain stable performance results. Possible values: *integer > 0*. [Default=5].
* **enable_log** - if False - no additional debug info ("silent"=1). If True ("verbosity"=3) it prints execution time by kernels. Possible values: *True, False*. [Default=False].
