# CUDA Accelerated Tree Construction Algorithm

## Usage
Specify the updater parameter as 'grow_gpu'. 

Python example:
```python
param['updater'] = 'grow_gpu'
```

## Dependencies
A CUDA capable GPU with at least compute capability >= 3.5 (the algorithm depends on shuffle and vote instructions introduced in Kepler).

The plugin also depends on CUB 1.5.4 - http://nvlabs.github.io/cub/index.html.

CUB is a header only cuda library which provides sort/reduce/scan primitives.


## Build
The plugin can be built using cmake and specifying the option PLUGIN_UPDATER_GPU=ON.

Specify the location of the CUB library with the cmake variable CUB_DIRECTORY.

It is recommended to build with Cuda Toolkit 7.5 or greater.

## Author
Rory Mitchell 

Report any bugs to r.a.mitchell.nz at google mail.


