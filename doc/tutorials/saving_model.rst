########################
Introduction to Model IO
########################

In XGBoost 1.0.0, we introduced experimental support of using `JSON
<https://www.json.org/json-en.html>`_ for saving/loading XGBoost models and related
hyper-parameters for training, aiming to replace the old binary internal format with an
open format that can be easily reused.  The support for binary format will be continued in
the future until JSON format is no-longer experimental and has satisfying performance.
This tutorial aims to share some basic insights into the JSON serialisation method used in
XGBoost.  Without explicitly mentioned, the following sections assume you are using the
experimental JSON format, which can be enabled by passing
``enable_experimental_json_serialization=True`` as training parameter, or provide the file
name with ``.json`` as file extension when saving/loading model:
``booster.save_model('model.json')``.  More details below.

Before we get started, XGBoost is a gradient boosting library with focus on tree model,
which means inside XGBoost, there are 2 distinct parts:

1. The model consisting of trees and
2. Hyperparameters and configurations used for building the model.

If you come from Deep Learning community, then it should be
clear to you that there are differences between the neural network structures composed of
weights with fixed tensor operations, and the optimizers (like RMSprop) used to train them.

So when one calls ``booster.save_model`` (``xgb.save`` in R), XGBoost saves the trees, some model
parameters like number of input columns in trained trees, and the objective function, which combined
to represent the concept of "model" in XGBoost.  As for why are we saving the objective as
part of model, that's because objective controls transformation of global bias (called
``base_score`` in XGBoost).  Users can share this model with others for prediction,
evaluation or continue the training with a different set of hyper-parameters etc.

However, this is not the end of story.  There are cases where we need to save something
more than just the model itself.  For example, in distrbuted training, XGBoost performs
checkpointing operation.  Or for some reasons, your favorite distributed computing
framework decide to copy the model from one worker to another and continue the training in
there.  In such cases, the serialisation output is required to contain enougth information
to continue previous training without user providing any parameters again.  We consider
such scenario as **memory snapshot** (or memory based serialisation method) and distinguish it
with normal model IO operation. Currently, memory snapshot is used in the following places:

* Python package: when the ``Booster`` object is pickled with the built-in ``pickle`` module.
* R package: when the ``xgb.Booster`` object is persisted with the built-in functions ``saveRDS``
  or ``save``.

Other language bindings are still working in progress.

.. note::

  The old binary format doesn't distinguish difference between model and raw memory
  serialisation format, it's a mix of everything, which is part of the reason why we want
  to replace it with a more robust serialisation method.  JVM Package has its own memory
  based serialisation methods.

To enable JSON format support for model IO (saving only the trees and objective), provide
a filename with ``.json`` as file extension:

.. code-block:: python
  :caption: Python

  bst.save_model('model_file_name.json')

.. code-block:: r
  :caption: R

  xgb.save(bst, 'model_file_name.json')

To use JSON to store memory snapshots, add ``enable_experimental_json_serialization`` as a training
parameter.  In Python this can be done by:

.. code-block:: python

  bst = xgboost.train({'enable_experimental_json_serialization': True}, dtrain)
  with open('filename', 'wb') as fd:
      pickle.dump(bst, fd)

Notice the ``filename`` is for Python intrinsic function ``open``, not for XGBoost.  Hence
parameter ``enable_experimental_json_serialization`` is required to enable JSON format.

Similarly, in the R package, add ``enable_experimental_json_serialization`` to the training
parameter:

.. code-block:: r

  params <- list(enable_experimental_json_serialization = TRUE, ...)
  bst <- xgboost.train(params, dtrain, nrounds = 10)
  saveRDS(bst, 'filename.rds')

***************************************************************
A note on backward compatibility of models and memory snapshots
***************************************************************

**We guarantee backward compatibility for models but not for memory snapshots.**

Models (trees and objective) use a stable representation, so that models produced in earlier
versions of XGBoost are accessible in later versions of XGBoost. **If you'd like to store or archive
your model for long-term storage, use** ``save_model`` (Python) and ``xgb.save`` (R).

On the other hand, memory snapshot (serialisation) captures many stuff internal to XGBoost, and its
format is not stable and is subject to frequent changes. Therefore, memory snapshot is suitable for
checkpointing only, where you persist the complete snapshot of the training configurations so that
you can recover robustly from possible failures and resume the training process. Loading memory
snapshot generated by an earlier version of XGBoost may result in errors or undefined behaviors.
**If a model is persisted with** ``pickle.dump`` (Python) or ``saveRDS`` (R), **then the model may
not be accessible in later versions of XGBoost.**

***************************
Custom objective and metric
***************************

XGBoost accepts user provided objective and metric functions as an extension.  These
functions are not saved in model file as they are language dependent feature.  With
Python, user can pickle the model to include these functions in saved binary.  One
drawback is, the output from pickle is not a stable serialization format and doesn't work
on different Python version or XGBoost version, not to mention different language
environment.  Another way to workaround this limitation is to provide these functions
again after the model is loaded. If the customized function is useful, please consider
making a PR for implementing it inside XGBoost, this way we can have your functions
working with different language bindings.

******************************************************
Loading pickled file from different version of XGBoost
******************************************************

As noted, pickled model is neither portable nor stable, but in some cases the pickled
models are valuable.  One way to restore it in the future is to load it back with that
specific version of Python and XGBoost, export the model by calling `save_model`.  To help
easing the mitigation, we created a simple script for converting pickled XGBoost 0.90
Scikit-Learn interface object to XGBoost 1.0.0 native model.  Please note that the script
suits simple use cases, and it's advised not to use pickle when stability is needed.
It's located in ``xgboost/doc/python`` with the name ``convert_090to100.py``.  See
comments in the script for more details.

A similar procedure may be used to recover the model persisted in an old RDS file. In R, you are
able to install an older version of XGBoost using the ``remotes`` package:

.. code-block:: r

  library(remotes)
  remotes::install_version("xgboost", "0.90.0.1")  # Install version 0.90.0.1

Once the desired version is installed, you can load the RDS file with ``readRDS`` and recover the
``xgb.Booster`` object. Then call ``xgb.save`` to export the model using the stable representation.
Now you should be able to use the model in the latest version of XGBoost.

********************************************************
Saving and Loading the internal parameters configuration
********************************************************

XGBoost's ``C API``, ``Python API`` and ``R API`` support saving and loading the internal
configuration directly as a JSON string.  In Python package:

.. code-block:: python

  bst = xgboost.train(...)
  config = bst.save_config()
  print(config)


or in R:

.. code-block:: R

  config <- xgb.config(bst)
  print(config)

Will print out something similiar to (not actual output as it's too long for demonstration):

.. code-block:: javascript

    {
      "Learner": {
        "generic_parameter": {
          "enable_experimental_json_serialization": "0",
          "gpu_id": "0",
          "gpu_page_size": "0",
          "n_jobs": "0",
          "random_state": "0",
          "seed": "0",
          "seed_per_iteration": "0"
        },
        "gradient_booster": {
          "gbtree_train_param": {
            "num_parallel_tree": "1",
            "predictor": "gpu_predictor",
            "process_type": "default",
            "tree_method": "gpu_hist",
            "updater": "grow_gpu_hist",
            "updater_seq": "grow_gpu_hist"
          },
          "name": "gbtree",
          "updater": {
            "grow_gpu_hist": {
              "gpu_hist_train_param": {
                "debug_synchronize": "0",
                "gpu_batch_nrows": "0",
                "single_precision_histogram": "0"
              },
              "train_param": {
                "alpha": "0",
                "cache_opt": "1",
                "colsample_bylevel": "1",
                "colsample_bynode": "1",
                "colsample_bytree": "1",
                "default_direction": "learn",

                ...

                "subsample": "1"
              }
            }
          }
        },
        "learner_train_param": {
          "booster": "gbtree",
          "disable_default_eval_metric": "0",
          "dsplit": "auto",
          "objective": "reg:squarederror"
        },
        "metrics": [],
        "objective": {
          "name": "reg:squarederror",
          "reg_loss_param": {
            "scale_pos_weight": "1"
          }
        }
      },
      "version": [1, 0, 0]
    }


You can load it back to the model generated by same version of XGBoost by:

.. code-block:: python

  bst.load_config(config)

This way users can study the internal representation more closely.  Please note that some
JSON generators make use of locale dependent floating point serialization methods, which
is not supported by XGBoost.

*************************************************
Difference between saving model and dumping model
*************************************************

XGBoost has a function called ``dump_model`` in Booster object, which lets you to export
the model in a readable format like ``text``, ``json`` or ``dot`` (graphviz).  The primary
use case for it is for model interpretation or visualization, and is not supposed to be
loaded back to XGBoost.  The JSON version has a `schema
<https://github.com/dmlc/xgboost/blob/master/doc/dump.schema>`_.  See next section for
more info.

***********
JSON Schema
***********

Another important feature of JSON format is a documented `Schema
<https://json-schema.org/>`_, based on which one can easily reuse the output model from
XGBoost.  Here is the initial draft of JSON schema for the output model (not
serialization, which will not be stable as noted above).  It's subject to change due to
the beta status.  For an example of parsing XGBoost tree model, see ``/demo/json-model``.
Please notice the "weight_drop" field used in "dart" booster.  XGBoost does not scale tree
leaf directly, instead it saves the weights as a separated array.

.. include:: ../model.schema
   :code: json

************
Future Plans
************

Right now using the JSON format incurs longer serialisation time, we have been working on
optimizing the JSON implementation to close the gap between binary format and JSON format.
