# XGBoost4J-Spark

(3 - 4 sentences intro)

# Build an Application with XGBoost4J-Spark

(based on maven build), step by step (structure of program)

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



### Current Version of Gang Scheduling

based on spark even listener

### Checkpoint Support

## Prediction

Highlight the recommended way (batching prediction)

briefly talk about single-instance prediction

## Model Persistence 

(also talk about how to train a model in Spark and use it in python environment) 

# Building a ML Pipeline with XGBoost4J-Spark







