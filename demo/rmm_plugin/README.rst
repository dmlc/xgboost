Using XGBoost with RAPIDS Memory Manager (RMM) plugin
=====================================================

`RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__ library provides a
collection of efficient memory allocators for NVIDIA GPUs. It is now possible to use
XGBoost with memory allocators provided by RMM, by enabling the RMM integration plugin.

The demos in this directory highlights one RMM allocator in particular: **the pool
sub-allocator**.  This allocator addresses the slow speed of ``cudaMalloc()`` by
allocating a large chunk of memory upfront. Subsequent allocations will draw from the pool
of already allocated memory and thus avoid the overhead of calling ``cudaMalloc()``
directly. See `this GTC talk slides
<https://on-demand.gputechconf.com/gtc/2015/presentation/S5530-Stephen-Jones.pdf>`_ for
more details.

Before running the demos, ensure that XGBoost is compiled with the RMM plugin enabled. To do this,
run CMake with option ``-DPLUGIN_RMM=ON`` (``-DUSE_CUDA=ON`` also required):

.. code-block:: sh

  cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DPLUGIN_RMM=ON
  make -j$(nproc)

CMake will attempt to locate the RMM library in your build environment. You may choose to build
RMM from the source, or install it using the Conda package manager. If CMake cannot find RMM, you
should specify the location of RMM with the CMake prefix:

.. code-block:: sh

  # If using Conda:
  cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DPLUGIN_RMM=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
  # If using RMM installed with a custom location
  cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DPLUGIN_RMM=ON -DCMAKE_PREFIX_PATH=/path/to/rmm

********************************
Informing XGBoost about RMM pool
********************************

When XGBoost is compiled with RMM, most of the large size allocation will go through RMM
allocators, but some small allocations in performance critical areas are using a different
caching allocator so that we can have better control over memory allocation behavior.
Users can override this behavior and force the use of rmm for all allocations by setting
the global configuration ``use_rmm``:

.. code-block:: python

  with xgb.config_context(use_rmm=True):
    clf = xgb.XGBClassifier(tree_method="hist", device="cuda")

Depending on the choice of memory pool size and the type of the allocator, this can add
more consistency to memory usage but with slightly degraded performance impact.

*******************************
No Device Ordinal for Multi-GPU
*******************************

Since with RMM the memory pool is pre-allocated on a specific device, changing the CUDA
device ordinal in XGBoost can result in memory error ``cudaErrorIllegalAddress``. Use the
``CUDA_VISIBLE_DEVICES`` environment variable instead of the ``device="cuda:1"`` parameter
for selecting device. For distributed training, the distributed computing frameworks like
``dask-cuda`` are responsible for device management.