Text Input Format of DMatrix
============================

## Basic Input Format
As we have mentioned, XGBoost takes LibSVM format. For training or predicting, XGBoost takes an instance file with the format as below:

train.txt
```
1 101:1.2 102:0.03
0 1:2.1 10001:300 10002:400
0 0:1.3 1:0.3
1 0:0.01 1:0.3
0 0:0.2 1:0.3
```
Each line represent a single instance, and in the first line '1' is the instance label,'101' and '102' are feature indices, '1.2' and '0.03' are feature values. In the binary classification case, '1' is used to indicate positive samples, and '0' is used to indicate negative samples. We also support probability values in [0,1] as label, to indicate the probability of the instance being positive.

Additional Information
----------------------
Note: these additional information are only applicable to single machine version of the package.

### Group Input Format
As XGBoost supports accomplishing [ranking task](../demo/rank), we support the group input format. In ranking task, instances are categorized into different groups in real world scenarios, for example, in the learning to rank web pages scenario, the web page instances are grouped by their queries. Except the instance file mentioned in the group input format, XGBoost need an file indicating the group information. For example, if the instance file is the "train.txt" shown above,
and the group file is as below:

train.txt.group
```
2
3
```
This means that, the data set contains 5 instances, and the first two instances are in a group and the other three are in another group. The numbers in the group file are actually indicating the number of instances in each group in the instance file in order.
While configuration, you do not have to indicate the path of the group file. If the instance file name is "xxx", XGBoost will check whether there is a file named "xxx.group" in the same directory and decides whether to read the data as group input format.

### Instance Weight File
XGBoost supports providing each instance an weight to differentiate the importance of instances. For example, if we provide an instance weight file for the "train.txt" file in the example as below:

train.txt.weight
```
1
0.5
0.5
1
0.5
```
It means that XGBoost will emphasize more on the first and fourth instance， that is to say positive instances while training.
The configuration is similar to configuring the group information. If the instance file name is "xxx", XGBoost will check whether there is a file named "xxx.weight" in the same directory and if there is, will use the weights while training models. Weights will be included into an "xxx.buffer" file that is created by XGBoost automatically. If you want to update the weights, you need to delete the "xxx.buffer" file prior to launching XGBoost.

### Initial Margin file
XGBoost supports providing each instance an initial margin prediction. For example, if we have a initial prediction using logistic regression for "train.txt" file, we can create the following file:

train.txt.base_margin
```
-0.4
1.0
3.4
```
XGBoost will take these values as initial margin prediction and boost from that. An important note about base_margin is that it should be margin prediction before transformation, so if you are doing logistic loss, you will need to put in value before logistic transformation. If you are using XGBoost predictor, use pred_margin=1 to output margin values.
