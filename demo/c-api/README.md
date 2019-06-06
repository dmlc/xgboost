C-APIs
===

**XGBoost** implements a C API originally designed for various language
bindings.  For detailed reference, please check xgboost/c_api.h.  Here is a
demonstration of using the API.

# Train
This example shows how to load data into a DMatrix, train and make predictions.

To run the training example from this directory:
```bash
cd train
make
# Make sure the system can find the xgboost shared library
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH=}:${PWD}/../../../lib
./c-api-demo
```
This demo assumes a Linux system with make and that the xgboost library is compiled.

# Custom DMatrix creation
This example shows how an external library can create a DMatrix object using function callbacks.

# CMake
If you use **CMake** for your project, you can either install **XGBoost**
somewhere in your system and tell CMake to find it by calling
`find_package(xgboost)`, or put **XGBoost** inside your project's source tree
and call **CMake** command: `add_subdirectory(xgboost)`.  To use
`find_package()`, put the following in your **CMakeLists.txt**:

``` CMake
find_package(xgboost REQUIRED)
add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo xgboost::xgboost)
```

If you want to put XGBoost inside your project (like git submodule), use this
instead:
``` CMake
add_subdirectory(xgboost)
add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo xgboost)
```

# make
You can start by modifying the makefile in this directory to fit your need.