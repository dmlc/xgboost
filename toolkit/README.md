Toolkit
====
This folder contains some example toolkits developed with rabit to help you get started. 

KMeans
====

#### Input File Format
KMeans uses LIBSVM format to parse the input. If you are not familiar with LIBSVM, <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/">here</a> you will find more details. 

The format is the following:

&lt;label&gt; &lt;index1&gt;:&lt;value1&gt; &lt;index2&gt;:&lt;value2&gt; ...

where label is a dummy integer value in this case (you can add 1's to every example), index&lt;x&gt; is the index for feature x, and value&lt;x&gt; is the feature x value.

#### Output File Format
KMeans currently outputs the centroids as dense vectors. Each line in the output file corresponds to a centroid. The number of lines in the file must match the number of clusters K you specified in the command line.

#### Example

Let's go over a more detailed example...

#  Preprocess

Download the smallwiki dataset used in the Machine Learning for Big Data class at University of Washington.

http://courses.cs.washington.edu/courses/cse547/14wi/datasets/smallwiki.zip

Unzip it, you should find three files:
* tfidf.txt: each row is in the form of “docid||termid1:tfidf1,termid2:tfidf2,...
* dictionary.txt: map of term to termid
* cluster0.txt: initial cluster centers. Won't needed.

The first thing to do is to convert the tfidf file format into the input format rabit supports, i.e. LIBSVM. For that, you can use a simple python script. The following should suffice. You should redirect the output to a file, let's say tfidf.libsvm.

```python
  for line in open("tfidf.txt").read().splitlines():
    example = line.split('|')[1].split(',')
    example = ' '.join(example)
    print '%s %s' % (1, example)
```
#  Compile

You will then need to build the KMeans program with ```make```, which will produce three binaries:

* kmeans.mpi: runs on MPI.
* kmeans.mock: uses a mock to simulate error conditions for testing purposes.
* kmeans.rabit: uses our C++ implementation.

#  Running with Hadoop
 
If you want to run it with Hadoop, you can execute the [./kmeans_hadoop.sh](./kmeans_hadoop.sh) script from your master node in cluster. 
You will have to edit the file in order to specify the path to the Hadoop Streaming jar. Afterwards, you can execute it with the following arguments (in the exact same order):

* number of worker nodes in your Hadoop cluster (i.e. number of slave nodes)
* path to the input data (HDFS path where you put the preprocessed file in libsvm format)
* number of clusters K (let's use 20 for this example)
* number of iterations to perform (let's use just 5 iterations)
* output path (HDFS path where to store the output data, must be a non-existent folder)

The current implementation runs for the amount of iterations you specify in the command line argument. If you would like to add some convergence criteria (e.g. when no cluster assignment changes between iterations you stop or something like that) you will have to modify [./kmeans.cc](./kmeans.cc). We leave that as an exercise to the reader :)

You may have noticed that [./kmeans_hadoop.sh](./kmeans_hadoop.sh) uses kmeans.rabit binary, but you can also use kmeans.mock in order to easily test your system behavior in presence of failures. More on that later.

Don't forget to copy the preprocessed file into HDFS and create the output folder. For example, inside the bin folder in Hadoop, you can execute the following:

```bash
$ ./hadoop fs -mkdir kmeans
$ ./hadoop fs -mkdir kmeans/in
$ ./hadoop fs -put tfidf.libsvm kmeans/in
$ ./hadoop fs -mkdir kmeans/out
```

#  Running with MPI

You will need to have a MPI cluster installed, for example OpenMPI. In order to run the program, you can use mpirun to submit the job. This is a non-fault tolerant version as it is backed by MPI.


#  Running with Mock

As previously mentioned, you can execute the kmeans example, an any of your own, with the mock binary. This will allow you to test error conditions while you are developing your algorithms. As explained in the [Tutorial](../guide), passing the script certain parameters (e.g. mock=0,0,1,0) will cause certain node to exit after calling Allreduce/Broadcast in some iteration.

You can also run this locally, you will only need to split the input file into several smaller files, each will be used by a particular process in the shared memory environment. You can use some Unix command line tool such as split.


#  Processing Output

Once the program finishes running, you can fetch the output from HDFS. For example, inside the bin folder in Hadoop, you can execute the following:

```bash
$ ./hadoop fs -get kmeans/out/part-00000 kmeans.out

```

Each line of the output file is a centroid in dense format. As this dataset contains the words in dictionary.txt file, you can do some simple post processing to recover the top 10 words of each centroid. Something like this should work:

```python
  words = {}
  for line in open("dictionary.txt").read().splitlines():
    word, index = line.split(' ')
    words[int(index)] = word
  
  from collections import defaultdict
  clusters = defaultdict(list)
  cluster_name = 0
  for line in open("kmeans.out").read().splitlines():
    line = line.split(' ')
    clusters[cluster_name].extend(line)
    cluster_name+=1

  import numpy as np
  for j, key in enumerate(clusters):
    elements = clusters[key]
    array = np.array(elements).astype(np.float32)
    idx = np.argsort(array)[::-1][:10]
    ws = []
    for i in idx:
      ws.append(words[i])
    print 'cluster %d = %s' % (j, ' '.join(ws))
```








