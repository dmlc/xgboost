############################
Text Input Format of DMatrix
############################

******************
Basic Input Format
******************
XGBoost currently supports two text formats for ingesting data: LibSVM and CSV. The rest of this document will describe the LibSVM format. (See `this Wikipedia article <https://en.wikipedia.org/wiki/Comma-separated_values>`_ for a description of the CSV format.)

For training or predicting, XGBoost takes an instance file with the format as below:

.. code-block:: none
  :caption: ``train.txt``

  1 101:1.2 102:0.03
  0 1:2.1 10001:300 10002:400
  0 0:1.3 1:0.3
  1 0:0.01 1:0.3
  0 0:0.2 1:0.3

Each line represent a single instance, and in the first line '1' is the instance label, '101' and '102' are feature indices, '1.2' and '0.03' are feature values. In the binary classification case, '1' is used to indicate positive samples, and '0' is used to indicate negative samples. We also support probability values in [0,1] as label, to indicate the probability of the instance being positive.

******************************************
Auxiliary Files for Additional Information
******************************************
**Note: all information below is applicable only to single-node version of the package.** If you'd like to perform distributed training with multiple nodes, skip to the section `Embedding additional information inside LibSVM file`_.

Group Input Format
==================
For `ranking task <https://github.com/dmlc/xgboost/tree/master/demo/rank>`_, XGBoost supports the group input format. In ranking task, instances are categorized into *query groups* in real world scenarios. For example, in the learning to rank web pages scenario, the web page instances are grouped by their queries. XGBoost requires an file that indicates the group information. For example, if the instance file is the ``train.txt`` shown above,  the group file should be named ``train.txt.group`` and be of the following format:

.. code-block:: none
  :caption: ``train.txt.group``

  2
  3

This means that, the data set contains 5 instances, and the first two instances are in a group and the other three are in another group. The numbers in the group file are actually indicating the number of instances in each group in the instance file in order.
At the time of configuration, you do not have to indicate the path of the group file. If the instance file name is ``xxx``, XGBoost will check whether there is a file named ``xxx.group`` in the same directory.

Instance Weight File
====================
Instances in the training data may be assigned weights to differentiate relative importance among them. For example, if we provide an instance weight file for the ``train.txt`` file in the example as below:

.. code-block:: none
  :caption: ``train.txt.weight``

  1
  0.5
  0.5
  1
  0.5

It means that XGBoost will emphasize more on the first and fourth instance (i.e. the positive instances) while training.
The configuration is similar to configuring the group information. If the instance file name is ``xxx``, XGBoost will look for a file named ``xxx.weight`` in the same directory. If the file exists, the instance weights will be extracted and used at the time of training.

.. note:: Binary buffer format and instance weights

  If you choose to save the training data as a binary buffer (using :py:meth:`save_binary() <xgboost.DMatrix.save_binary>`), keep in mind that the resulting binary buffer file will include the instance weights. To update the weights, use the :py:meth:`set_weight() <xgboost.DMatrix.set_weight>` function.

Initial Margin File
===================
XGBoost supports providing each instance an initial margin prediction. For example, if we have a initial prediction using logistic regression for ``train.txt`` file, we can create the following file:

.. code-block:: none
  :caption: ``train.txt.base_margin``

  -0.4
  1.0
  3.4

XGBoost will take these values as initial margin prediction and boost from that. An important note about base_margin is that it should be margin prediction before transformation, so if you are doing logistic loss, you will need to put in value before logistic transformation. If you are using XGBoost predictor, use ``pred_margin=1`` to output margin values.

***************************************************
Embedding additional information inside LibSVM file
***************************************************
**This section is applicable to both single- and multiple-node settings.**

Query ID Columns
================
This is most useful for `ranking task <https://github.com/dmlc/xgboost/tree/master/demo/rank>`_, where the instances are grouped into query groups. You may embed query group ID for each instance in the LibSVM file by adding a token of form ``qid:xx`` in each row:

.. code-block:: none
  :caption: ``train.txt``

  1 qid:1 101:1.2 102:0.03
  0 qid:1 1:2.1 10001:300 10002:400
  0 qid:2 0:1.3 1:0.3
  1 qid:2 0:0.01 1:0.3
  0 qid:3 0:0.2 1:0.3
  1 qid:3 3:-0.1 10:-0.3
  0 qid:3 6:0.2 10:0.15

Keep in mind the following restrictions:

* You are not allowed to specify query ID's for some instances but not for others. Either every row is assigned query ID's or none at all.
* The rows have to be sorted in ascending order by the query IDs. So, for instance, you may not have one row having large query ID than any of the following rows.

Instance weights
================
You may specify instance weights in the LibSVM file by appending each instance label with the corresponding weight in the form of ``[label]:[weight]``, as shown by the following example:

.. code-block:: none
  :caption: ``train.txt``

  1:1.0 101:1.2 102:0.03
  0:0.5 1:2.1 10001:300 10002:400
  0:0.5 0:1.3 1:0.3
  1:1.0 0:0.01 1:0.3
  0:0.5 0:0.2 1:0.3

where the negative instances are assigned half weights compared to the positive instances.
