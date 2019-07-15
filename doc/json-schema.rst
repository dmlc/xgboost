################################
XGBoost JSON Schema, Version 1.0
################################

Preface
=======
This document contains an exhaustive description of the XGBoost JSON schema, a
mapping between XGBoost object classes and JSON objects and arrays. We aim to
store a **complete representation** of all XGBoost objects.

Representation of ``dmlc::Parameter`` objects
---------------------------------------------
Every object of subclasses of ``dmlc::Parameter`` are to be represented as JSON
objects. We create a bridge method to seamlessly convert ``dmlc::Parameter`` objects into
JSON objects, such that proper value types are used.

For example, consider the parameter class

.. code-block:: cpp

  struct MyParam : public dmlc::Parameter<MyParam> {
    float learning_rate;
    int num_hidden;
    int activation;
    std::string name;
    DMLC_DECLARE_PARAMETER(MyParam) {
      DMLC_DECLARE_FIELD(num_hidden);
      DMLC_DECLARE_FIELD(learning_rate);
      DMLC_DECLARE_FIELD(activation).add_enum("relu", 1).add_enum("sigmoid", 2);
      DMLC_DECLARE_FIELD(name);
    }
  };

and an object created by the following initialization code:

.. code-block:: cpp

  MyParam param;
  param.learning_rate = 0.1f;
  param.num_hidden = 10;
  param.activation = 2;  // sigmoid
  param.name = "MyNet";

This collection is naturally expressed as the following JSON object:

.. parsed-literal::

  {
    "learning_rate": "0.1",
    "num_hidden": "10",
    "activation": "sigmoid",
    "name": "MyNet"
  }

Notice the value fields are strings even for numeric values.

Notations
---------

In the following sections, the schema for each XGBoost class is shown as a JSON
object. Fields whose keys are marked with *italic* are optional and may be
absent in some models. The hyper-linked value indicate that the value shall be
the JSON representation of another XGBoost class. The *italic* value indicate
that the value shall be a primitive type (string, integer, floating-point etc).
Every mention of ``floating-point`` refers to single-precision floating point
(32-bit), unless explicitly stated otherwise.  Every mention of ``integer``
refers to 32-bit integer unless stated otherwise.

Full content of the schema
==========================

Note: Click :ref:`here <example>` for a minimal example of the current schema.

.. contents:: :local:

XGBoostModel
------------
This is the root object for XGBoost model.

.. parsed-literal::

  {
    "version" : ["1", "0"],
    "learner" : Learner_
  }

Learner
-------
.. parsed-literal::

  {
    "gradient_booster" : GradientBooster_,
    "learner_train_param" : LearnerTrainParam_,
    "learner_model_param": LearnerModelParam_,
    "metrics" : [ *array of* Metric_ ],
    "objective" : Objective_
  }

The ``learner_train_param`` field stores (hyper)parameters used for training, while
``learner_model_param`` field stores model related properties.

The ``gradient_booster`` field stores an gradient boosted ensemble consisting of
models of certain type (e.g. tree, linear).

The ``metrics`` field is used to store evaluation metrics.

The ``objective`` field stores the objective (loss) function used to train the
ensemble model.

LearnerTrainParam
-----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "dsplit": *string*,
    "disable_default_eval_metric": *boolean*,
    "booster": "string",
    "objective": "string",
    "num_boosting_round" : *integer*
  }

The ``dsplit`` field indicates the data partitioning mode for distributed
learning. Its value should be one of ``auto``, ``col``, and ``row``. The value
should be set to ``auto`` when only a single node is used for training.

The ``num_boosting_round`` field stores the number of boosting rounds performed.
This number is different from the number of trees if ``num_parallel_tree`` of
GBTreeTrainParam_ is greater than 1.

LearnerModelParam
-----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

   {
     "base_score": "*integer*",
     "num_feature" : "*integer*",
     *"num_class"* : "*integer*"
   }


The ``num_class`` is used only for multi-class classification task, in which it
indicates the number of output classes.

GenericParameter
----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "seed": "*integer*",
    "seed_per_iteration": "*boolean*",
    "nthread": "*integer*"
    *"gpu_id"*: "*integer*",
    *"n_gpus"*: "*integer*"
  }

The ``gpu_id`` and ``n_gpus`` fields are used to set the GPU(s) to use for
training and prediction. If no GPU is used, the fields should be omitted.

GradientBooster
---------------
Currently, we may choose one of the three subclasses for the gradient boosted
ensemble:

* GBTree_: decision tree models
* Dart_: DART (Dropouts meet Multiple Additive Regression Trees) models
* GBLinear_: linear models

We can determine which subclass was used by looking at the ``name`` field
of each subclass.

Metric
------
.. parsed-literal::

  *string*

For the time being, every metric is fully specified by a single string. In the
future, we may want to add extra parameters to some metrics. When that happens,
we should add subclasses of ``Metric``.

The string must be a valid metric name as specified by the `parameter
doc <https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters>`_.

Objective
---------
Currently, we may choose one of the 10 subclasses for the objective function:

* RegLossObj_
* SoftmaxMultiClassObj_
* HingeObj_
* LambdaRankObj_
* PairwiseRankObj_
* LambdaRankObjNDCG_
* LambdaRankObjMAP_
* PoissonRegression_
* CoxRegression_
* GammaRegression_
* TweedieRegression_

GBTree
------
The ``GBTree`` class stores an ensemble of decision trees that are produced
via gradient boosting. It is a subclass of GradientBooster_.

.. parsed-literal::

  {
    "name" : "gbtree",
    "gbtree_train_param" : GBTreeTrainParam_,
    "model" : GBTreeModel_
  }

The ``gbtree_train_param`` field is the list of training parameters specific to
``GBTree``.

GBTreeTrainParam
----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_parallel_tree": *integer*,
    "predictor": *string*,
    "process_type": *string*,
    "tree_method": string,
    "updater": string,
  }

The ``num_parallel_tree`` field denotes the number of parallel trees constructed
during each iteration. It is used to support boosted random forest.

The ``updater`` field stores a comma separated list of updater names that was provided at
the beginning of training. This field may not necessarily match the sequence given in the
``updaters`` field of GBTree_ or Dart_ due to auto configuration.

The ``process_type`` field denotes whether to create new trees (``default``) or
to update existing trees (``update``) during the boosting process. The field's
value must be either ``default`` or ``update``. Keep in mind that ``update`` is
highly experimental; most use cases will use ``default``.

The ``tree_method`` field is the choice of tree construction and its value should be one
of ``auto``, ``approx``, ``exact``, ``hist``, ``gpu_exact (deprecated)``, and
``gpu_hist``. The value should be set to ``auto`` when the base learner is not a decision
tree (e.g. linear model).

Dart
----
The ``Dart`` class stores an ensemble of decision trees that are produced
via gradient boosting, with dropouts at training time. This class is a subclass
of GBTree_ and hence contains all fields that GBTree_ contains. It is a subclass
of GradientBooster_.

.. parsed-literal::

  {
    "name" : "Dart",
    "dart_train_param" : DartTrainParam_,
    "model" : GBTreeModel_,
    *"weight_drop"* : [ *array of floating-point* ]
  }

This class has ``dart_train_param`` as the set of training parameters
specific to ``Dart``.

The ``weight_drop`` field stores the weights assigned to individual trees.
The weights should be used at training time.

DartTrainParam
--------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "sample_type": *string*,
    "normalize_type": *string*,
    "rate_drop": *floating-point*,
    "one_drop": *boolean*,
    "skip_drop": *floating-point*,
    "learning_rate": *floating-point*
  }

The meaning of these parameters is to be found in `the parameter doc
<https://xgboost.readthedocs.io/en/latest/parameter.html#additional-parameters-for-dart-booster-booster-dart>`_.

The ``sample_type`` field must be either ``uniform`` or ``weighted``.

The ``normalize_type`` field must be either ``tree`` or ``forest``.

GBTreeModel
-----------
The ``GBTreeModel`` class is the list of regression trees, plus the model
parameters.

.. parsed-literal::

  {
    "model_param" : GBTreeModelParam_,
    "trees" : [ *array of* RegTree_ ],
    *"tree_info"* : [ *array of integer* ]
  }

``tree_info`` is a reserved field, retained for the sake of compatibility with
the current binary serialization method.

GBTreeModelParam
----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_trees": *integer*,
    "num_feature" : *integer*,
    "num_output_group" : *integer*
  }

The ``num_output_group`` is the size of prediction per instance. This value is
set to 1 for all tasks except multi-class classification. For multi-class
classification, ``num_output_group`` must be set to the number of classes. This
must be identical to the value for ``num_class`` field of LearnerTrainParam_.

Note. ``num_roots`` and ``size_leaf_vector`` have been omitted due to
deprecation.

RegTree
-------
.. parsed-literal::

  {
    "tree_param" : TreeParam_,
    "nodes" : [ *array of* Node_ ],
    "stats" : [ *array of* NodeStat_ ]
  }

The first node in the ``nodes`` array specify root node.

The ``nodes`` array specify an adjacency list for an acyclic directed binary
tree graph. Each tree node has zero or two outgoing edges and exactly one
incoming edge. Cycles are not allowed.

TreeParam
---------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_nodes": "*integer*",
    "num_roots": "*integer*",
    *"num_deleted"* : *integer*
    "num_feature": *integer*
  }

The ``num_deleted`` field is optional and indicates that some node IDs are
marked deleted and thus should be re-used for creating new nodes. This exists
since the pruning method leaves gaps in node IDs. When omitted, ``num_deleted``
is assumed to be zero. This field may be deprecated in the future.

Note. ``num_roots`` and ``size_leaf_vector`` have been omitted due to
deprecation. ``max_depth`` is removed because it is not used anywhere in the
codebase.

Node
--------
Each node is represented as a JSON array of a fixed size, each element
storing the following fields:

.. parsed-literal::

  [
    *integer* (child_left_id),
    *integer* (child_right_id),
    *integer* (parent_id)
    *unsigned integer* (feature_id),
    *floating-point* (threshold)
    *boolean* (default_left)
  ]

The ``feature_id`` and ``threshold`` fields specify the feature ID and threshold used in
the test node, where the test is of form ``data[feature_id] < threshold``.  The
``child_left_id`` and ``child_right_id`` fields specify the nodes to be taken in a tree
traversal when the test ``data[feature_id] < threshold`` is true and false,
respectively. When ``child_left_id`` equals -1, current node is a leaf.  The node IDs are
0-based offsets to the ``nodes`` arrays in RegTree_. The ``default_left`` field indicates
the default direction in a tree traversal when feature value for ``feature_id`` is
missing.

NodeStat
--------
Statistics for each node is represented as a JSON array of a fixed size, each
element storing the following fields:

.. parsed-literal::

  [
    *floating-point* (loss_chg),
    *floating-point* (sum_hess),
    *floating-point* (base_weight),
    *integer* (leaf_child_cnt)
  ]

Note. ``leaf_child_cnt`` may be omitted in the future because it is only internally used
by the tree pruner. For serialization / deserialization, ``leaf_child_cnt`` should always
be set to 0.

GBLinear
--------
The ``GBLinear`` class stores an ensemble of linear models that are produced
via gradient boosting. It is a subclass of GradientBooster_.

.. parsed-literal::

  {
    "name" : "GBLinear",
    "gblinear_train_param" : GBLinearTrainParam_,
    "model": GBLinearModel_
  }

The ``num_boosting_round`` field stores the number of boosting rounds performed.

GBLinearTrainParam
------------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "updater" : *string*,
    "tolerance" : *floating-point*
  }

The ``updater`` field is the name of the linear updater used for training. Its
value must match that of ``updater`` in GBLinear_.

The ``tolerance`` field is the threshold for early stopping, in which iterations
were terminated if the largest weight updater is smaller than the threshold.
Setting it to zero disables early stopping.

Note. ``max_row_perbatch`` is omitted because it is deprecated.

GBLinearModel
-------------

.. parsed-literal::

  {
    "model_param" : GBLinearModelParam_,
    "weight" : [ *array of floating-point* ]
  }

The ``weight`` field stores the final coefficients of the combined linear model,
after all boosting rounds. Currently, the linear booster does not store
coefficients of individual boosting rounds.

GBLinearModelParam
------------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_feature" : *integer*,
    "num_output_group" : *integer*
  }

The ``num_output_group`` is the size of prediction per instance. This value is
set to 1 for all tasks except multi-class classification. For multi-class
classification, ``num_output_group`` must be set to the number of classes.

SoftmaxMultiClassObj
--------------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "SoftmaxMultiClassObj",
    "num_class" : *integer*,
    "output_prob" : *boolean*
  }

The ``num_class`` field must have the same value as ``num_class`` in
LearnerTrainParam_.

The ``output_prob`` field determines whether the loss function should produce
class index (``false``) or probability distribution (``true``).

HingeObj
--------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "HingeObj"
  }

RegLossObj
----------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "RegLossObj",
    "loss_type" : *string*,
    "scale_pos_weight": *floating-point*
  }

The ``loss_type`` field must be one of the following: ``LinearSquareLoss``,
``LogisticRegression``, ``LogisticClassification`` and ``LogisticRaw``.

LambdaRankObj
-------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "LambdaRankObj",
    "num_pairsample": *integer*,
    "fix_list_weight": *floating-point*
  }

The ``num_pairsample`` specifies the number of pairs to sample (per instance)
to compute the pairwise ranking loss.

The ``fix_list_weight`` field is the normalization factor for the weight of
each query group. If set to 0, it has no effect.

PairwiseRankObj
---------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "PairwiseRankObj",
    "num_pairsample": *integer*,
    "fix_list_weight": *floating-point*
  }

The ``num_pairsample`` specifies the number of pairs to sample (per instance)
to compute the pairwise ranking loss.

The ``fix_list_weight`` field is the normalization factor for the weight of
each query group. If set to 0, it has no effect.

LambdaRankObjNDCG
-----------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "LambdaRankObjNDCG",
    "num_pairsample": *integer*,
    "fix_list_weight": *floating-point*
  }

LambdaRankObjMAP
----------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "LambdaRankObjMAP",
    "num_pairsample": *integer*,
    "fix_list_weight": *floating-point*
  }

PoissonRegression
-----------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "PoissonRegression",
    "max_delta_step": *floating-point*
  }

CoxRegression
-------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "CoxRegression"
  }

GammaRegression
---------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "GammaRegression"
  }

TweedieRegression
-----------------
This class is a subclass of Objective_.

.. parsed-literal::

  {
    "name" : "TweedieRegression",
    "tweedie_variance_power" : *floaing-point*
  }

.. _example:

Minimal example
===============

.. code-block:: json

  {
    "version" : ["1", "0"],
    "learner" : {
      "learner_train_param" : {
        "seed": "0",
        "dsplit": "auto",
        "disable_default_eval_metric": false,
	"booster": "gbtree",
	"objective": "reg:squaredloss"
      },
      "gradient_booster" : {
        "name" : "GBTree",
        "num_boosting_round" : 2,
        "gbtree_train_param" : {
          "num_parallel_tree" : 1,
          "updater_seq" : [ "grow_quantile_histmaker" ],
          "process_type" : "default",
          "predictor" : "cpu_predictor"
        },
        "model" : {
          "model_param" : {
            "num_trees" : 2,
            "num_feature" : 126,
            "num_output_group" : 1
          },
          "trees" : [
            {
              "tree_param" : {
                "num_nodes": 9,
                "num_feature" : 126
              },
              "nodes" : [
                [1, 2,  28,  0.0,  true],
                [3, 4,  55,  0.5, false],
                [5, 6, 108,  1.0,  true],
                 1.8,
                -1.9,
                [7, 8,  66, -0.5,  true],
                 1.87,
                -1.99,
                 0.94
              ],
              "stats" : [
                [200.0, 1635.2,  0.2, 4000],
                [150.2,  922.8,  1.1, 2200],
                [300.4,  712.5, -1.5, 1800],
                [  0.0,  808.3,  0.0, 2000],
                [  0.0,  114.5,  0.0,  200],
                [100.1,  698.0, -1.8, 1600],
                [  0.0,   14.5,  0.0,  200],
                [  0.0,  686.8,  0.0, 1500],
                [  0.0,   11.2,  0.0,  100]
              ]
            },
            {
              "tree_param" : {
                "num_nodes": 3,
                "num_feature" : 126
              },
              "nodes" : [
                [1, 2, 5, 0.5, false],
                 1.0,
                -1.0
              ],
              "stats" : [
                [335.0, 135.2,  0.6, 4000],
                [  0.0,  88.3,  0.0, 3000],
                [  0.0,  46.9,  0.0, 1000]
              ]
            }
          ]
        }
      },
      "eval_metrics" : [ "auc" ],
      "objective" : {
        "name" : "RegLossObj",
        "loss_type" : "LogisticClassification",
        "scale_pos_weight": 1.0
      }
    }
  }
