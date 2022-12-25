########################
Introduction to Model IO
########################

In XGBoost 1.0.0, we introduced support of using `JSON
<https://www.json.org/json-en.html>`_ for saving/loading XGBoost models and related
hyper-parameters for training, aiming to replace the old binary internal format with an
open format that can be easily reused.  Later in XGBoost 1.6.0, additional support for
`Universal Binary JSON <https://ubjson.org/>`__ is added as an optimization for more
efficient model IO.  They have the same document structure with different representations,
and we will refer them collectively as the JSON format. This tutorial aims to share some
basic insights into the JSON serialisation method used in XGBoost.  Without explicitly
mentioned, the following sections assume you are using the one of the 2 outputs formats,
which can be enabled by providing the file name with ``.json`` (or ``.ubj`` for binary
JSON) as file extension when saving/loading model: ``booster.save_model('model.json')``.
More details below.

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
more than just the model itself.  For example, in distributed training, XGBoost performs
checkpointing operation.  Or for some reasons, your favorite distributed computing
framework decide to copy the model from one worker to another and continue the training in
there.  In such cases, the serialisation output is required to contain enough information
to continue previous training without user providing any parameters again.  We consider
such scenario as **memory snapshot** (or memory based serialisation method) and distinguish it
with normal model IO operation. Currently, memory snapshot is used in the following places:

* Python package: when the ``Booster`` object is pickled with the built-in ``pickle`` module.
* R package: when the ``xgb.Booster`` object is persisted with the built-in functions ``saveRDS``
  or ``save``.
* JVM packages: when the ``Booster`` object is serialized with the built-in functions ``saveModel``.

Other language bindings are still working in progress.

.. note::

  The old binary format doesn't distinguish difference between model and raw memory
  serialisation format, it's a mix of everything, which is part of the reason why we want
  to replace it with a more robust serialisation method.  JVM Package has its own memory
  based serialisation methods.

To enable JSON format support for model IO (saving only the trees and objective), provide
a filename with ``.json`` or ``.ubj`` as file extension, the latter is the extension for
`Universal Binary JSON <https://ubjson.org/>`__

.. code-block:: python
  :caption: Python

  bst.save_model('model_file_name.json')

.. code-block:: r
  :caption: R

  xgb.save(bst, 'model_file_name.json')

.. code-block:: Scala
  :caption: Scala

  val format = "json"  // or val format = "ubj"
  model.write.option("format", format).save("model_directory_path")

.. note::

  Only load models from JSON files that were produced by XGBoost. Attempting to load
  JSON files that were produced by an external source may lead to undefined behaviors
  and crashes.

While for memory snapshot, UBJSON is the default starting with xgboost 1.6.

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
functions are not saved in model file as they are language dependent features.  With
Python, user can pickle the model to include these functions in saved binary.  One
drawback is, the output from pickle is not a stable serialization format and doesn't work
on different Python version nor XGBoost version, not to mention different language
environments.  Another way to workaround this limitation is to provide these functions
again after the model is loaded. If the customized function is useful, please consider
making a PR for implementing it inside XGBoost, this way we can have your functions
working with different language bindings.

******************************************************
Loading pickled file from different version of XGBoost
******************************************************

As noted, pickled model is neither portable nor stable, but in some cases the pickled
models are valuable.  One way to restore it in the future is to load it back with that
specific version of Python and XGBoost, export the model by calling `save_model`.

A similar procedure may be used to recover the model persisted in an old RDS file. In R,
you are able to install an older version of XGBoost using the ``remotes`` package:

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

Will print out something similar to (not actual output as it's too long for demonstration):

.. code-block:: javascript

    {
      "Learner": {
        "generic_parameter": {
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
<https://github.com/dmlc/xgboost/blob/master/doc/dump.schema>`__.  See next section for
more info.

***********
JSON Schema
***********

Another important feature of JSON format is a documented `schema
<https://json-schema.org/>`__, based on which one can easily reuse the output model from
XGBoost.  Here is the JSON schema for the output model (not serialization, which will not
be stable as noted above).  For an example of parsing XGBoost tree model, see
``/demo/json-model``.  Please notice the "weight_drop" field used in "dart" booster.
XGBoost does not scale tree leaf directly, instead it saves the weights as a separated
array.

.. include:: ../model.schema
   :code: json
