############################
XGBoost Internal Feature Map
############################

The following is a reference to the features supported by XGBoost.  It is not a beginner's guide, but rather a list meant to help those looking to add new features to XGBoost understand what needs to be covered.

*************
Core Features
*************
Core features are not dependent on language binding and any language binding can choose to support them.

-------------
Data Storage
-------------
The primary data structure in XGBoost for storing user inputs is ``DMatrix``; it's a container for all data that XGBoost can use. ``QuantileDMatrix`` is a variant specifically designed for the ``hist`` tree method. Both can take GPU-based inputs. They take an optional parameter ``missing`` to specify which input value should be ignored. For external memory support, please refer to :doc:`/tutorials/external_memory`.

---------------------
Single Node Training
---------------------
There are two different model types in XGBoost: the tree model, which we primarily focus on, and the linear model. For the tree model, we have various methods to build decision trees; please see the :doc:`/treemethod` for a complete reference. In addition to the tree method, we have many hyper-parameters for tuning the model and injecting prior knowledge into the training process. Two noteworthy examples are :doc:`monotonic constraints </tutorials/monotonic>` and :doc:`feature interaction constraints </tutorials/feature_interaction_constraint>`. These two constraints require special treatment during tree construction. Both the ``hist`` and the ``approx`` tree methods support GPU acceleration. Also, XGBoost GPU supports gradient-based sampling, which supports external-memory data as well.

The objective function plays an important role in training. It not only provides the gradient, but also responsible for estimating a good starting point for Newton optimization. Please note that users can define custom objective functions for the task at hand.
In addition to numerical features, XGBoost also supports categorical features with two different algorithms, including one-hot encoding and optimal partitioning. For more information, refer to the :doc:`categorical feature tutorial </tutorials/categorical>`. The ``hist`` and the ``approx`` tree methods support categorical features for CPU and GPU.

There's working-in-progress support for vector leaves, which are decision tree leaves that contain multiple values. This type of tree is used to support efficient multi-class and multi-target models.

----------
Inference
----------
By inference, we specifically mean getting model prediction for the response variable. XGBoost supports two inference methods. The first one is the prediction on the ``DMatrix`` object (or ``QuantileDMatrix``, which is a subclass). Using a ``DMatrix`` object allows XGBoost to cache the prediction, hence getting faster performance when running prediction on the same data with new trees. The second method is ``inplace_predict``, which bypasses the construction of ``DMatrix``. It's more efficient but doesn't support cached prediction. In addtion to returning the estimated response, we also support returning the leaf index, which can be used to analyse the model and as a feature to another model.

----------
Model IO
----------
We have a set of methods for different model serialization methods, including complete serialization, saving to a file, and saving to a buffer. For more, refer to the :doc:`/tutorials/saving_model`.

-------------------
Model Explanation
-------------------
XGBoost includes features designed to improve understanding of the model. Here's a list:

- Global feature importance.
- SHAP value, including contribution and intervention.
- Tree dump.
- Tree visualization.
- Tree as dataframe.

For GPU support, the SHAP value uses the `GPUTreeShap <https://github.com/rapidsai/gputreeshap/tree/main>`_ project in rapidsai. They all support categorical features, while vector-leaf is still in progress.

----------
Evaluation
----------
XGBoost has built-in support for a wide range of metrics, from basic regression to learning to rank and survival modeling. They can handle distributed training and GPU-based acceleration. Custom metrics are supported as well, please see :doc:`/tutorials/custom_metric_obj`.

--------------------
Distributed Training
--------------------
XGBoost has built-in support for three distributed frameworks, including ``Dask``, ``PySpark``, and ``Spark (Scala)``. In addition, there's ``flink`` support for the Java binding and the ``ray-xgboost`` project. Please see the respective tutorial on how to use them. By default, XGBoost uses sample-based parallelism for distributed training. The column-based split is still working in progress and needs to be supported in these high-level framework integrations. On top of distributed training, we are also working on federated learning for both sample-based and column-based splits.

Distributed training works with custom objective functions and metrics as well. XGBoost aggregates the evaluation result automatically during training.

The distributed training is enabled by a built-in implementation of a collective library. It's based on the RABIT project and has evolved significantly since its early adoption. The collective implementation supports GPU via NCCL, and has variants for handling federated learning and federated learning on GPU.

Inference normally doesn't require any special treatment since we are using sample-based split. However, with column-based data split, we need to initialize the communicator context as well.

*****************
Language Bindings
*****************
We have a list of bindings for various languages. Inside the XGBoost repository, there's Python, R, Java, Scala, and C. All language bindings are built on top of the C version. Some others, like Julia and Rust, have their own repository. For guideline on adding a new binding, please see :doc:`/contrib/consistency`.