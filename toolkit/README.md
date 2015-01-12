Toolkit
====
This folder contains some example toolkits developed with rabit to help you get started. 

KMeans
====

#### How to run it
You will need to build the program with ```make```. 
If you want to run it with Hadoop, you can execute the [./kmeans_hadoop.sh](./kmeans_hadoop.sh) script from your master node in cluster. 
You will have to edit the file in order to specify the path to the Hadoop Streaming jar. Afterwards, you can execute it with the following arguments (in the exact same order):

1) number of worker nodes in your Hadoop cluster (i.e. number of slaves)
2) path to the input data (HDFS path where you put the data)
3) number of clusters K
4) number of iterations to perform
5) output path (HDFS path where to store the output data, must be new)

If you take a look at [./kmeans_hadoop.sh](./kmeans_hadoop.sh), you can see that it runs the kmeans.rabit version. If you want to run the program backed by the mock, you will need to update it accordingly, i.e. use kmeans.mock instead.

The current implementation runs for the amount of iterations you specify in the command line argument. If you would like to add some convergence criteria (e.g. when no cluster assignment changes between iterations you stop or something like that) you will have to modify [./kmeans.cc](./kmeans.cc). We leave that as an exercise to the reader :)

#### Input File Format
KMeans uses LIBSVM format to parse the input. If you are not familiar with LIBSVM, [http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/](here) you will find more details. 

The format is the following:

<label> <index1>:<value1> <index2>:<value2> ...
.
.
.

where label is a dummy integer value in this case (you can add 1's to every example), index<x> is the index for feature x, and value<x> is the feature x value.

#### Output File Format
KMeans currently outputs the centroids as dense vectors. Each line in the output file corresponds to a centroid. The number of lines in the file must match the number of clusters K you specified in the command line.
