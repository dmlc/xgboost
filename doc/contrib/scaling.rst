#######
Scaling
#######

************
What is this
************

This document contains some notes for the those who wish to customize XGBoost or make
feature improvements to push the scaling limit. It's more of a developer guide instead of
a user tutorial. If you are a beginner or have no issue training XGBoost models on your
machine, you can safely ignore this document.

Data is the blood and the backbone of machine learning models. There's no ultimate
solution for fighting the battle of balancing variance and bias. Indeed, we have more and
more innovative machine learning algorithms and efficient hyper-parameter tuning
strategies. They help us to see what we see today: a miracle enabled by modern machine
learning algorithms. However, at the end of the day, it's the data that matters the
most. Machine learning models are mere compression of what we can observe from the vast
amount of data.

Training gradient boosting models on large datasets is no small feet. When a dataset is
tiny, the training procedure is straightforward: Prepare your data, create a
``scikit-learn`` compatible estimator like the :py:class:`~xgboost.XGBClassifier`, put it
into an HPO library like `optuna` or just a grid-search class like
:py:class:`sklearn.model_selection.GridSearchCV`, fetch a coffee and wait for the good
news. The day is bright, and life is easy. Beginners can do all these on their
laptops. Once the data size exceeds certain limits, things start to get messy. Your model
takes forever to train, your computer is running out of memory, and your Jupyter notebook
might crash without warning.

To scale XGBoost beyond one's desktop and adapt to big modern data, we must employ more
intelligent algorithms and utilize resources beyond a single CPU. We have been trying to
push the limit on two different but related fronts: the computation demand and the memory
usage. This document gives a high-level overview of what we have been working on, focusing
on memory usage. We will include references to various documents for specific sets of
features.

****************
GPU Acceleration
****************

GPU excels at handling mass amounts of data thanks to its parallel computation
model. XGBoost supports CUDA-based GPU acceleration and can consume data hosted on GPU
memory end-to-end with an almost complete implementation using CUDA. Details about using
GPU usage can be found at :doc:`/gpu/index`, with an introduction to using GPU
clusters. We will touch on distributed training in the following sections. The way in
which we use GPU has evolved, and XGBoost is still trying to catch up. GPU itself has
evolved from a simple accelerator to a system-level solution. Numerous innovations happen
on the GPU-focus system design with a large ecosystem under heavy construction. There is
`GPUDirect` connects network storage directly to PCIe and consumes data from RAID-0
arrays. There are new architectures like the grace-hopper that combine arm-based CPUs and
GPUs. There are new distributed frameworks like Legate designed with GPU in mind. Last but
not least, we have tensor cores specifically designed for approximated matrix
computation. XGBoost utilizes only the most commonly seen features in a GPU and has yet to
get a chance to explore these new frontiers. We believe there is plenty of room to grow.

*************
Data Iterator
*************

One profound limitation of gradient boosting with the Newton method is that it's not a
stochastic algorithm. A user must train the model using the entire dataset instead of
feeding the model with small batches. There are many novel ideas for ameliorating this
issue (like the gradient-based sampling mentioned at the end of this document), but none
gets to a point like what deep learning does. We can not boost each tree using only 32
samples and expect the final model to generalize to billions of data. Not all hope is
lost, though. Users can define a custom iterator to load data in chunks for running
XGBoost algorithms. The custom iterator is one of the essential building blocks for
scaling XGBoost beyond a single computer's host memory. It serves multiple use cases:

- `QuantileDMatrix`
- Distributed Framework Integration
- External Memory

All these features are to address the scaling problem. The intuition behind it is using
chunks of data instead of loading a giant blob in memory. The
:py:class:`~xgboost.QuantileDmatrix` can load a dataset with an iterator and compress the
data chunk-by-chunk based on a quantile sketching algorithm before concatenation. As for
distributed training, when a distributed framework like `PySpark` or `Dask` is employed,
the data is usually divided into chunks or partitions. Lastly, the external memory version
of :py:class:`~xgboost.DMatrix` can fetch data from a cache stored on a fast external
memory (like an NVMe hard drive) during training and inference without any concatenation
unless GPU is used (GPU uses a hybrid approach, see more in
:doc:`/tutorials/external_memory`).

You can find a quick introduction to defining a custom iterator at
:doc:`/tutorials/external_memory`. Passing the iterator to the
:py:class:`~xgboost.QuantileDMatrix` enables direct construction of
:py:class:`~xgboost.QuantileDMatrix` with data chunks. On the other hand, if it's passed
to :py:class:`~xgboost.DMatrix`, it enables the external memory feature instead. The
Python interface sits at a higher level of abstraction. For a brief introduction to the
underlying C API, you can look at :ref:`c_streaming`. Under the hood, it's implemented as
a set of callback functions that allows XGBoost to fetch data from user-defined
sources. These callbacks are only used during the construction of `DMatrix` and discarded
during training. Once the construction of DMatrix is finished, users can safely free all
the related data.

****************
QuantilelDMatrix
****************

One can use the :py:class:`~xgboost.QuantileDMatrix` with a custom iterator if the input
is comparable to the available memory size. Typically, a user would pass the data like
:py:class:`pandas.DataFrame` to the `QuantileDMatrix`. However, we can use an iterator if
the available memory cannot fit the dataframe and the `QuantileDmatrix` together.

The :py:class:`~xgboost.QuantileDMatrix` is an optimization for the ``hist`` and
``gpu_hist`` tree method. When the tree method allows, it's used internally in
scikit-learn compatible estimators like :py:class:`~xgboost.XGBRegressor`. It consumes the
input data like :py:class:`numpy.ndarray` or :py:class:`cudf.DataFrame` directly without
creating an intermediate data structures like the :py:class:`~xgboost.DMatrix`. The
iterator interface enables this optimization.

For most of the users out there, they may not notice the iterator used under the
hood. However, if your dataset has already taken most of the memory space with little room
left for XGBoost, one can offload the dataset onto disk (or host memory if you are using
GPU), fetch them by batches to construct the :py:class:`~xgboost.QuantileDMatrix`. This
way, one can ease the memory pressure by not loading the raw input into memory. The
:py:class:`~xgboost.QuantileDMatrix` still occupies a non-trivial fraction of memory of
the original size of the raw input. But with most of the raw input offloaded onto an
external memory, you can significantly lower the peak memory usage. For an example usage,
please see :ref:`sphx_glr_python_examples_quantile_data_iterator.py`

Unsurprisingly, the data iterator was first introduced exclusively for GPU training with
:py:class:`~xgboost.QuantileDMatrix`, where the memory is most constrained. We generalized
the interface to use it in other parts of XGBoost. Using an iterator, users can offload
the memory to either host memory or external devices during the construction of
:py:class:`~xgboost.QuantileDMatrix`. Additionally, since data is divided into batches,
the memory pressure for sorting the input data for quantile sketching is significantly
reduced. Efficient parallel sorting algorithms require a double buffer for storing the
results, which is fine for most cases until one wants to sort a machine learning
dataset. The expectation that if one has 16GB of memory, they should be able to use 16GB
of data is often assumed but challenging to achieve in practice. Using the
:py:class:`~xgboost.QuantileDMatrix` can help us get close to and sometimes even exceed
that expectation. Internally, the GPU algorithm uses a compressed ELLPACK page (the
`QuantileDMatrix`) to represent that data. Thanks to the compression, the size of the
ELLPACK is usually smaller than the input data. Combined with other runtime data, XGBoost
should be able to train a dataset on GPU with a size similar to the GPU memory.

********************
Distributed Training
********************

If using a single device doesn't get the job done, using more devices can
help. Distributed training is considered the most efficient and practical way to scale
XGBoost. It works for CPU and GPU clusters, and scales with memory and computation
power. Since distributed computing is the most widely adopted solution for many other
types of problems, everyone can access a large pool of workers with ease by using public
cloud infrastructures like AWS, GKE or NGC. XGBoost has integration with multiple
distributed computing frameworks, including but not limited to:

- Dask
- PySpark
- Spark
- Ray

All of them can use GPU for efficient training. There are introductions of each
integration in :doc:`/tutorials/index`. Beyond these frameworks, users can create custom
distributed training procedures using facilities inside XGBoost like the
:py:class:`~xgboost.RabitTracker`. These framework integrations can use
:py:class:`~xgboost.DataIter` under the hood to save memory when appropriate. It's not yet
default for all of them, but we are getting there.

Distributed frameworks usually divide data into chunks or partitions for efficient data
processing and scattering. This data pattern fits naturally with the data iterator
design. Using the Dask interface as an example. When a user calls XGBoost to train models
on a dask DataFrame, what XGBoost does is first fetch all partitions in each worker (but
without moving any of them between workers), iterate through these partitions using a
custom :py:class:`~xgboost.DataIter` with the help of
:py:class:`~xgboost.QuantileDMatrix`. It could be a better solution as it's challenging to
incorporate the data-spilling capability of the framework, but it's a massive improvement
over the previous approach. In before, XGBoost has to first concatenate all partitions
into one, then create a `DMatrix` object from the giant blob before creating a
histogram-based representation. Two to three copies of the data are made during the
process, and they have to sit in the host memory together. As a matter of fact, two or
three copies is an optimistic estimation. Python libraries hide much of the complexity for
a friendly user experience. Since the :py:class:`~xgboost.QuantileDMatrix` (which is built
upon the :py:class:`~xgboost.DataIter`) can consume a wide range of primitive data types,
there's no need for a user to cast data to float. A simple `astype(np.float32)` operation
might well crush the memory as it creates yet another copy of data.

****************************
External Memory and Sampling
****************************

The custom iterator also enables the external memory feature. See
:doc:`/tutorials/external_memory` for an introduction, and the :ref:`ext_remarks` for what
can be expected from using external memory.

One optimization worth repeating here is using external memory with gradient-based
sampling.  With uniform sampling, setting the sampling rate below :math:`0.8` might make a
den into the model accuracy. However, with gradient-based sampling, the rate can be as low
as :math:`0.2` without much impact on the final output. The GPU implementation utilizes
this algorithm for efficient external memory training.