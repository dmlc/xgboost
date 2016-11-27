Distributed XGBoost YARN on AWS
===============================
This is a step-by-step tutorial on how to setup and run distributed [XGBoost](https://github.com/dmlc/xgboost)
on a AWS EC2 cluster. Distributed XGBoost runs on various platforms such as MPI, SGE and Hadoop YARN.
In this tutorial, we use YARN as an example since this is widely used solution for distributed computing.

Prerequisite
------------
We need to get a [AWS key-pair](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
to access the AWS services. Let us assume that we are using a key ```mykey``` and  the corresponding permission file ```mypem.pem```.

We also need [AWS credentials](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html),
which includes an `ACCESS_KEY_ID` and a `SECRET_ACCESS_KEY`.

Finally, we will need a S3 bucket to host the data and the model, ```s3://mybucket/```

Setup a Hadoop YARN Cluster
---------------------------
This sections shows how to start a Hadoop YARN cluster from scratch.
You can skip this step if you have already have one.
We will be using [yarn-ec2](https://github.com/tqchen/yarn-ec2) to start the cluster.

We can first clone the yarn-ec2 script by the following command.
```bash
git clone https://github.com/tqchen/yarn-ec2
```

To use the script, we must set the environment variables `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` properly. This can be done by adding the following two lines in
`~/.bashrc` (replacing the strings with the correct ones)

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

Now we can launch a master machine of the cluster from EC2
```bash
./yarn-ec2 -k mykey -i mypem.pem launch xgboost
```
Wait a few mininutes till the master machine get up.

After the master machine gets up, we can query the public DNS of the master machine using the following command.
```bash
./yarn-ec2 -k mykey -i mypem.pem get-master xgboost
```
It will show the public DNS of the master machine like ```ec2-xx-xx-xx.us-west-2.compute.amazonaws.com```
Now we can open the browser, and type(replace the DNS with the master DNS)
```
ec2-xx-xx-xx.us-west-2.compute.amazonaws.com:8088
```
This will show the job tracker of the YARN cluster. Note that we may wait a few minutes before the master finishes bootstrapping and starts the
job tracker.

After master machine gets up, we can freely add more slave machines to the cluster.
The following command add m3.xlarge instances to the cluster.
```bash
./yarn-ec2 -k mykey -i mypem.pem -t m3.xlarge -s 2 addslave xgboost
```
We can also choose to add two spot instances
```bash
./yarn-ec2 -k mykey -i mypem.pem -t m3.xlarge -s 2 addspot xgboost
```
The slave machines will startup, bootstrap  and report to the master.
You can check if the slave machines are connected by clicking on Nodes link on the job tracker.
Or simply type the following URL(replace DNS ith the master DNS)
```
ec2-xx-xx-xx.us-west-2.compute.amazonaws.com:8088/cluster/nodes
```

One thing we should note is that not all the links in the job tracker works.
This is due to that many of them uses the private ip of AWS, which can only be accessed by EC2.
We can use ssh proxy to access these packages.
Now that we have setup a cluster with one master and two slaves. We are ready to run the experiment.


Build XGBoost with S3
---------------------
We can log into the master machine by the following command.
```bash
./yarn-ec2 -k mykey -i mypem.pem login xgboost
```

We will be using S3 to host the data and the result model, so the data won't get lost after the cluster shutdown.
To do so, we will need to build xgboost with S3 support. The only thing we need to do is to set ```USE_S3```
variable to be true. This can be achieved by the following command.

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
cp make/config.mk config.mk
echo "USE_S3=1" >> config.mk
make -j4
```
Now we have built the XGBoost with S3 support. You can also enable HDFS support if you plan to store data on HDFS, by turnning on ```USE_HDFS``` option.

XGBoost also relies on the environment variable to access S3, so you will need to add the following two lines to `~/.bashrc` (replacing the strings with the correct ones)
on the master machine as well.

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export BUCKET=mybucket
```

Host the Data on S3
-------------------
In this example, we will copy the example dataset in xgboost to the S3 bucket as input.
In normal usecases, the dataset is usually created from existing distributed processing pipeline.
We can use [s3cmd](http://s3tools.org/s3cmd) to copy the data into mybucket(replace ${BUCKET} with the real bucket name).

```bash
cd xgboost
s3cmd put demo/data/agaricus.txt.train s3://${BUCKET}/xgb-demo/train/
s3cmd put demo/data/agaricus.txt.test s3://${BUCKET}/xgb-demo/test/
```

Submit the Jobs
---------------
Now everything is ready, we can submit the xgboost distributed job to the YARN cluster.
We will use the [dmlc-submit](https://github.com/dmlc/dmlc-core/tree/master/tracker) script to submit the job.

Now we can run the following script in the distributed training folder(replace ${BUCKET} with the real bucket name)
```bash
cd xgboost/demo/distributed-training
# Use dmlc-submit to submit the job.
../../dmlc-core/tracker/dmlc-submit --cluster=yarn --num-workers=2 --worker-cores=2\
    ../../xgboost mushroom.aws.conf nthread=2\
    data=s3://${BUCKET}/xgb-demo/train\
    eval[test]=s3://${BUCKET}/xgb-demo/test\
    model_dir=s3://${BUCKET}/xgb-demo/model
```
All the configurations such as ```data``` and ```model_dir``` can also be directly written into the configuration file.
Note that we only specified the folder path to the file, instead of the file name.
XGBoost will read in all the files in that folder as the training and evaluation data.

In this command, we are using two workers, each worker uses two running thread.
XGBoost can benefit from using multiple cores in each worker.
A common choice of working cores can range from 4 to 8.
The trained model will be saved into the specified model folder. You can browse the model folder.
```
s3cmd ls s3://${BUCKET}/xgb-demo/model/
```

The following is an example output from distributed training.
```
16/02/26 05:41:59 INFO dmlc.Client: jobname=DMLC[nworker=2]:xgboost,username=ubuntu
16/02/26 05:41:59 INFO dmlc.Client: Submitting application application_1456461717456_0015
16/02/26 05:41:59 INFO impl.YarnClientImpl: Submitted application application_1456461717456_0015
2016-02-26 05:42:05,230 INFO @tracker All of 2 nodes getting started
2016-02-26 05:42:14,027 INFO [05:42:14] [0]  test-error:0.016139        train-error:0.014433
2016-02-26 05:42:14,186 INFO [05:42:14] [1]  test-error:0.000000        train-error:0.001228
2016-02-26 05:42:14,947 INFO @tracker All nodes finishes job
2016-02-26 05:42:14,948 INFO @tracker 9.71754479408 secs between node start and job finish
Application application_1456461717456_0015 finished with state FINISHED at 1456465335961
```

Analyze the Model
-----------------
After the model is trained, we can analyse the learnt model and use it for future prediction task.
XGBoost is a portable framework, the model in all platforms are ***exchangeable***.
This means we can load the trained model in python/R/Julia and take benefit of data science pipelines
in these languages to do model analysis and prediction.

For example, you can use [this ipython notebook](https://github.com/dmlc/xgboost/tree/master/demo/distributed-training/plot_model.ipynb)
to plot feature importance and visualize the learnt model.

Trouble Shooting
----------------

When you encountered a problem, the best way might be use the following command
to get logs of stdout and stderr of the containers, to check what causes the problem.
```
yarn logs -applicationId yourAppId
```

Future Directions
-----------------
You have learnt to use distributed XGBoost on YARN in this tutorial.
XGBoost is portable and scalable framework for gradient boosting.
You can checkout more examples and resources in the [resources page](https://github.com/dmlc/xgboost/blob/master/demo/README.md).

The project goal is to make the best scalable machine learning solution available to all platforms.
The API is designed to be able to portable, and the same code can also run on other platforms such as MPI and SGE.
XGBoost is actively evolving and we are working on even more exciting features
such as distributed xgboost python/R package. Checkout [RoadMap](https://github.com/dmlc/xgboost/issues/873) for
more details and you are more than welcomed to contribute to the project.
