# CUDA Accelerated Tree Construction Algorithms
This plugin adds GPU accelerated tree construction algorithms to XGBoost.
## Usage
Specify the 'updater' parameter as one of the following algorithms. 
updater | Description
--- | ---
grow_gpu | The standard XGBoost tree construction algorithm. Performs exact search for splits. Slower and uses considerably more memory than 'grow_gpu_hist'
grow_gpu_hist | Equivalent to the XGBoost fast histogram algorithm. Faster and uses considerably less memory. Splits may be less accurate.

All algorithms currently use only a single GPU. The device ordinal can be selected using the 'gpu_id' parameter, which defaults to 0.

This plugin currently works with the CLI version and python version.

Python example:
```python
param['gpu_id'] = 1
param['updater'] = 'grow_gpu'
```
## Benchmarks

[See here](http://dmlc.ml/2016/12/14/GPU-accelerated-xgboost.html) for performance benchmarks of the 'grow_gpu' updater.


## Dependencies
A CUDA capable GPU with at least compute capability >= 3.5 (the algorithm depends on shuffle and vote instructions introduced in Kepler).

Building the plug-in requires CUDA Toolkit 7.5 or later.

The plugin also depends on CUB 1.6.4 - https://nvlabs.github.io/cub/

CUB is a header only cuda library which provides sort/reduce/scan primitives.


## Build
To use the plugin xgboost must be built using cmake specifying the option PLUGIN_UPDATER_GPU=ON. The location of the CUB library must also be specified with the cmake variable CUB_DIRECTORY. CMake will prepare a build system depending on which platform you are on.

From the command line on Windows or Linux starting from the xgboost directory:

```bash
$ mkdir build
$ cd build
$ cmake .. -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY=<MY_CUB_DIRECTORY>
```

On Windows you may also need to specify your generator as 64 bit, so the cmake command becomes:
```bash
$ cmake .. -G"Visual Studio 12 2013 Win64" -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY=<MY_CUB_DIRECTORY>
```
You may also  be able to use a later version of visual studio depending on whether the CUDA toolkit supports it.

On an linux cmake will generate a Makefile in the build directory. Invoking the command 'make' from this directory will build the project. If the build fails try invoking make again. There can sometimes be problems with the order items are built.

On Windows cmake will generate an xgboost.sln solution file in the build directory. Build this solution in release mode. This is also a good time to check it is being built as x64. If not make sure the cmake generator is set correctly.

The build process generates an xgboost library and executable as normal but containing the GPU tree construction algorithm.

## Changelog
##### 2017/4/25
* Add fast histogram algorithm
* Fix Linux build
* Add 'gpu_id' parameter

## References
[Mitchell, Rory, and Eibe Frank. Accelerating the XGBoost algorithm using GPU computing. No. e2911v1. PeerJ Preprints, 2017.](https://peerj.com/preprints/2911/)

## Author
Rory Mitchell 

Please report bugs to the xgboost/issues page. You can tag me with @RAMitchell.

Otherwise I can be contacted at r.a.mitchell.nz at gmail.


