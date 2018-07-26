# XGBoost4J-Spark

(3 - 4 sentences intro)

# Build an Application with XGBoost4J-Spark

To build a Spark application with XGBoost4J-Spark, you first need to refer to the dependency in maven_central, 

You can add the following dependency in your pom file.

```xml
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j-spark</artifactId>
    <version>latest_version_num</version>
</dependency>
```

For the latest release version number, please check [here](https://github.com/dmlc/xgboost/releases).

We also publish some functionalities which would be included in the coming release in the form of snapshot version. To access 
these functionalities, you can refer the dependency to snapshot artifacts. We publish snapshot version in github-based repo, so 
you first need to add the following repo in pom.xml:

```xml
<repository>
  <id>GitHub Repo</id>
  <name>GitHub Repo</name>
  <url>https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/</url>
</repository>
``` 

and then refer to the snapshot dependency by adding:

```xml
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j</artifactId>
    <version>next_version_num-SNAPSHOT</version>
</dependency>
```

## Data Preparation

As aforementioned, XGBoost4J-Spark seamlessly integrates Spark and XGBoost. The integration enables
 users to apply various types of transformation over the training/test datasets with the convenient 
 and powerful data processing framework, Spark.  
 
In this section, we use [Iris](https://archive.ics.uci.edu/ml/datasets/iris) dataset as an example to
 showcase how we use Spark to transform raw dataset and make it fit the requirement of XGBoost.

Iris dataset is shipped in CSV format. Each instance contains 4 features, "sepal length", "sepal width",
"petal length" and "petal width". "class" column in each instance is 
essentially the label which has three distinct values: "Iris Setosa", "Iris Versicolour" 
and "Iris Virginica". 

### Read Dataset with Spark Built-In CSV Reader

The first thing in data transformation is to load the dataset as Spark's structured data abstraction,
DataFrame.

```scala
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
    
    val spark = SparkSession.builder().getOrCreate()
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    val rawInput = spark.read.schema(schema).csv("input_path")
```

At the first line, we create a instance of [SparkSession](http://spark.apache.org/docs/latest/sql-programming-guide.html#starting-point-sparksession)
 which is the entry of any Spark program working with DataFrame. The `schema` variable defines the schema of DataFrame wrapping 
 Iris data from csv file. With this explicitly set schema, we can define the columns' name as well as their types. Finally, we can 
 use the built-in csv reader to load Iris csv file as a DataFrame named `rawInput`.
 
### Transform Raw Iris Dataset

To make Iris dataset be recognizable to XGBoost, we need to 

1. Transform String-typed label, i.e. "class", to Integer-typed label.

2. Assemble the feature columns as a vector to build XGBoost's internal data representation, DMatrix.

To convert String-typed label to Integer, we can use Spark's built-in feature transformer StringIndexer.

```scala
    import org.apache.spark.ml.feature.StringIndexer
    val stringIndexer = new StringIndexer().
      setInputCol("class").
      setOutputCol("classIndex").
      fit(rawInput)
    val labelTransformed = stringIndexer.transform(rawInput).drop("class")
``` 

To create a StringIndexer, we set input column, i.e. the column containing String-typed label, and output column, 
i.e. the column to contain the Integer-typed label. Then we `fit` StringIndex with our input DataFrame so that Spark internals can 
get information like total number of distinct values, etc. Now we have a StringIndexer ready to be applied to our input DataFrame.

To execute the transformation logic of StringIndexer, we `transform` the input DataFrame with the StringIndexer and to keep simplicity, 
we drop the column `class` which contains the original String-typed labels.

`fit` and `transform` are two key operations in MLLIB. Basically, `fit` produces a "transformer", e.g. StringIndexer, and each 
transformer apply `transform` method on dataset to add new column which contains transformed features/labels or prediction results, etc.
You can find more details in [here](http://spark.apache.org/docs/latest/ml-pipeline.html#pipeline-components).
      
Similarly, we can use another transformer, 'VectorAssembler', to assemble feature columns "sepal length", "sepal width", 
"petal length" and "petal width" as a vector.      

```scala
    import org.apache.spark.ml.feature.VectorAssembler
    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
      setOutputCol("features")
    val xgbInput = vectorAssembler.transform(labelTransformed).select("features",
      "classIndex")
```

Now, we have a DataFrame containing only two columns, "features" which contains vector-represented
"sepal length", "sepal width", "petal length" and "petal width"; and also "classIndex" which has Integer-typed
labels. This DataFrame can be feed to train a XGBoost model directly.
      
## Training 

XGBoost support both Regression and Classification. In this doc we use Iris dataset to show the usage of XGBoost 
in the case of multi-class Classification. The usage in Regression is very similar with Classification.

To train a XGBoost model for classification, we need to claim a XGBoostClassifier first:

```scala
    import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
    val xgbParam = Map("eta" -> 0.1f,
          "max_depth" -> 2,
          "objective" -> "multi:softprob",
          "num_class" -> 3,
          "num_round" -> 100,
          "num_workers" -> 2)
    val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("features").
          setLabelCol("classIndex")
```  

The available parameters for training a XGBoost model can be found in [here](https://xgboost.readthedocs.io/en/latest/parameter.html).
In XGBoost4J-Spark, we support not only the default set of parameters but also the camel-case-variance of these parameters to keep consistent
with Spark's MLLIB parameters. Specifically, each parameter in [here](https://xgboost.readthedocs.io/en/latest/parameter.html) has its 
equivalent form in XGBoost4J-Spark with camel case. For example, to set max_depth for each tree, you can pass parameter just like what we
 do in the above code snippet, or you can do it through setters in XGBoostClassifer:
 
 ```scala
     val xgbClassifier1 = new XGBoostClassifier().
       setFeaturesCol("features").
       setLabelCol("classIndex")
     xgbClassifier1.setMaxDeltaStep(2)
 ```

After we set XGBoostClassifier parameters and feature/label column, we can build a transformer, XGBoostClassificationModel, and apply
transformation to the DataFrame containing training set, i.e. xgbInput. 

```scala
    val xgbClassificationModel = xgbClassifier.fit(xgbInput)
    val results = xgbClassificationModel.transform(xgbInput)
```

Now, we get a DataFrame, result, containing margin, probability for each class and the prediction for each instance

```scala
+-----------------+----------+--------------------+--------------------+----------+
|         features|classIndex|       rawPrediction|         probability|prediction|
+-----------------+----------+--------------------+--------------------+----------+
|[5.1,3.5,1.4,0.2]|       0.0|[3.45569849014282...|[0.99579632282257...|       0.0|
|[4.9,3.0,1.4,0.2]|       0.0|[3.45569849014282...|[0.99618089199066...|       0.0|
|[4.7,3.2,1.3,0.2]|       0.0|[3.45569849014282...|[0.99643349647521...|       0.0|
|[4.6,3.1,1.5,0.2]|       0.0|[3.45569849014282...|[0.99636095762252...|       0.0|
|[5.0,3.6,1.4,0.2]|       0.0|[3.45569849014282...|[0.99579632282257...|       0.0|
|[5.4,3.9,1.7,0.4]|       0.0|[3.45569849014282...|[0.99428516626358...|       0.0|
|[4.6,3.4,1.4,0.3]|       0.0|[3.45569849014282...|[0.99643349647521...|       0.0|
|[5.0,3.4,1.5,0.2]|       0.0|[3.45569849014282...|[0.99579632282257...|       0.0|
|[4.4,2.9,1.4,0.2]|       0.0|[3.45569849014282...|[0.99618089199066...|       0.0|
|[4.9,3.1,1.5,0.1]|       0.0|[3.45569849014282...|[0.99636095762252...|       0.0|
|[5.4,3.7,1.5,0.2]|       0.0|[3.45569849014282...|[0.99428516626358...|       0.0|
|[4.8,3.4,1.6,0.2]|       0.0|[3.45569849014282...|[0.99643349647521...|       0.0|
|[4.8,3.0,1.4,0.1]|       0.0|[3.45569849014282...|[0.99618089199066...|       0.0|
|[4.3,3.0,1.1,0.1]|       0.0|[3.45569849014282...|[0.99618089199066...|       0.0|
|[5.8,4.0,1.2,0.2]|       0.0|[3.45569849014282...|[0.97809928655624...|       0.0|
|[5.7,4.4,1.5,0.4]|       0.0|[3.45569849014282...|[0.97809928655624...|       0.0|
|[5.4,3.9,1.3,0.4]|       0.0|[3.45569849014282...|[0.99428516626358...|       0.0|
|[5.1,3.5,1.4,0.3]|       0.0|[3.45569849014282...|[0.99579632282257...|       0.0|
|[5.7,3.8,1.7,0.3]|       0.0|[3.45569849014282...|[0.97809928655624...|       0.0|
|[5.1,3.8,1.5,0.3]|       0.0|[3.45569849014282...|[0.99579632282257...|       0.0|
+-----------------+----------+--------------------+--------------------+----------+

``` 

### Parallel/Distributed Training

One of the most important parameters we set for XGBoostClassifier is "num_workers" (or "numWorkers").
This parameter controls how many parallel workers we want to have when training a XGBoostClassificationModel.

In XGBoost4J-Spark, each XGBoost worker is wrapped by a Spark task. By default, we allocate a core per each XGBoost worker.
Therefore, the OpenMP optimization within each XGBoost worker does not take effect and the parallelization of training is achieved
 by running multiple workers (i.e. Spark tasks) at the same time. 
 
 If you do want OpenMP optimization, you have to 
 
 1. set `nthread` to a value larger than 1 when creating XGBoostClassifier/XGBoostRegressor
 
 2. set `spark.task.cpus` in Spark to the same value as `nthread`
 
### Run XGBoost4J-Spark in Production

XGBoost4J-Spark has attracted a lot of users from industry and is deployed in many production environments. We also include many features
enabling running XGBoost4J-Spark in production smoothly.
 
#### Gang Scheduling

XGBoost uses [AllReduce](http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/)
 to synchronize the stats of each worker. Therefore XGBoost4J-Spark requires that all of `nthread * numWorkers` cores
  should be available before the training runs.
  
However, in production environment where many users share the same cluster, it's hard to guarantee that your XGBoost4J-Spark can get
all requested resources for every run. By default, the communication layer in XGBoost will block the whole application when it requires more
cores to be available. This process usually brings unnecessary resource waste as it keeps the ready resources and try to claim more.
 Additionally, this usually happens silently and does not bring the attention of users.
 
 XGBoost4J-Spark allows the user to setup a timeout threshold for claiming resources from the cluster. If the application cannot get
 enough resources within this time period, the application would fail instead of wasting resources for hanging long. To enable this feature,
 you can set with XGBoostClassifier:
 
 ```scala
 xgbClassifier.setTimeoutRequestWorkers()
 ```
 
 or pass in `timeout_request_workers` in xgbParamMap when building XGBoostClassifier
 
 ```scala
    val xgbParam = Map("eta" -> 0.1f,
       "max_depth" -> 2,
       "objective" -> "multi:softprob",
       "num_class" -> 3,
       "num_round" -> 100,
       "num_workers" -> 2,
       "timeout_request_workers" -> 60000L)
    val xgbClassifier = new XGBoostClassifier(xgbParam).
        setFeaturesCol("features").
        setLabelCol("classIndex")
 ```

#### Checkpoint During Training

Transient Failures are commonly seen in production environment. To simplify the design of XGBoost,
 we stop training if any of the distributed workers fail.  Additionally, to efficiently recover failed training, we support
 checkpoint mechanism to facilitate failure recovery.
 
 To enable this feature, you can set how many iterations we build each checkpoint with `setCheckpointInterval` and
 the path store checkpointPath with `setCheckpointPath`:
 
  ```scala
      xgbClassifier.setCheckpointInterval(2)
      xgbClassifier.setCheckpointPath("/checkpoint_path")
  ```
  
  an equivalent way is to pass in parameters in XGBoostClassifier's constructor:
  
  ```scala
      val xgbParam = Map("eta" -> 0.1f,
         "max_depth" -> 2,
         "objective" -> "multi:softprob",
         "num_class" -> 3,
         "num_round" -> 100,
         "num_workers" -> 2,
         "checkpoint_path" -> "/checkpoints",
         "checkpoint_interval" -> 2)
      val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("features").
          setLabelCol("classIndex")
   ```

If the training failed during these 100 rounds, the next run of training would start by reading the latest checkpoint file 
in `/checkpoints` and start from the iteration when the checkpoint was built until to next failure or the specified 100 rounds.

## Prediction

Highlight the recommended way (batching prediction)

briefly talk about single-instance prediction

## Model Persistence 

(also talk about how to train a model in Spark and use it in python environment) 

# Building a ML Pipeline with XGBoost4J-Spark







