# CUDA Accelerated Tree Construction Algorithm

## Usage
Specify the updater parameter as 'grow_gpu'. The max depth of the algorithm is 6 - setting a max depth above this will cause the program to fail. 

Python example:
```python
param['updater'] = 'grow_gpu'
```

## Requirements
A CUDA capable GPU with at least compute capability >= 3.5 (the algorithm depends on shuffle and vote instructions introduced in Kepler).

## Build
The plugin can be built using cmake and specifying the option PLUGIN_UPDATER_GPU=ON.

It is recommended to build with Cuda Toolkit 7.5 or greater.

## Author
Rory Mitchell 

Report any bugs to r.a.mitchell.nz at google mail.


