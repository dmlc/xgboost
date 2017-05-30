XGBoost4J Java API
==================
This tutorial introduces

## Data Interface
Like the xgboost python module, xgboost4j uses ```DMatrix``` to handle data,
libsvm txt format file, sparse matrix in CSR/CSC format, and dense matrix is
supported.

* To import ```DMatrix``` :
```java
import org.dmlc.xgboost4j.DMatrix;
```

* To load libsvm text format file, the usage is like :
```java
DMatrix dmat = new DMatrix("train.svm.txt");
```

* To load sparse matrix in CSR/CSC format is a little complicated, the usage is like :
suppose a sparse matrix :
1 0 2 0
4 0 0 3
3 1 2 0

  for CSR format
```java
long[] rowHeaders = new long[] {0,2,4,7};
float[] data = new float[] {1f,2f,4f,3f,3f,1f,2f};
int[] colIndex = new int[] {0,2,0,3,0,1,2};
DMatrix dmat = new DMatrix(rowHeaders, colIndex, data, DMatrix.SparseType.CSR);
```

  for CSC format
```java
long[] colHeaders = new long[] {0,3,4,6,7};
float[] data = new float[] {1f,4f,3f,1f,2f,2f,3f};
int[] rowIndex = new int[] {0,1,2,2,0,2,1};
DMatrix dmat = new DMatrix(colHeaders, rowIndex, data, DMatrix.SparseType.CSC);
```

* To load 3*2 dense matrix, the usage is like :
suppose a matrix :
1    2
3    4
5    6

```java
float[] data = new float[] {1f,2f,3f,4f,5f,6f};
int nrow = 3;
int ncol = 2;
float missing = 0.0f;
DMatrix dmat = new Matrix(data, nrow, ncol, missing);
```

* To set weight :
```java
float[] weights = new float[] {1f,2f,1f};
dmat.setWeight(weights);
```

## Setting Parameters
* in xgboost4j any ```Iterable<Entry<String, Object>>``` object could be used as parameters.

* to set parameters, for non-multiple value params, you can simply use entrySet of an Map:
```java
Map<String, Object> paramMap = new HashMap<>() {
  {
    put("eta", 1.0);
    put("max_depth", 2);
    put("silent", 1);
    put("objective", "binary:logistic");
    put("eval_metric", "logloss");
  }
};
Iterable<Entry<String, Object>> params = paramMap.entrySet();
```
* for the situation that multiple values with same param key, List<Entry<String, Object>> would be a good choice, e.g. :
```java
List<Entry<String, Object>> params = new ArrayList<Entry<String, Object>>() {
    {
        add(new SimpleEntry<String, Object>("eta", 1.0));
        add(new SimpleEntry<String, Object>("max_depth", 2.0));
        add(new SimpleEntry<String, Object>("silent", 1));
        add(new SimpleEntry<String, Object>("objective", "binary:logistic"));
    }
};
```

## Training Model
With parameters and data, you are able to train a booster model.
* Import ```Trainer``` and ```Booster``` :
```java
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.util.Trainer;
```

* Training
```java
DMatrix trainMat = new DMatrix("train.svm.txt");
DMatrix validMat = new DMatrix("valid.svm.txt");
//specify a watchList to see the performance
//any Iterable<Entry<String, DMatrix>> object could be used as watchList
List<Entry<String, DMatrix>> watchs =  new ArrayList<>();
watchs.add(new SimpleEntry<>("train", trainMat));
watchs.add(new SimpleEntry<>("test", testMat));
int round = 2;
Booster booster = Trainer.train(params, trainMat, round, watchs, null, null);
```

* Saving model
After training, you can save model and dump it out.
```java
booster.saveModel("model.bin");
```

* Dump Model and Feature Map
```java
booster.dumpModel("modelInfo.txt", false)
//dump with featureMap
booster.dumpModel("modelInfo.txt", "featureMap.txt", false)
```

* Load a model
```java
Params param = new Params() {
  {
    put("silent", 1);
    put("nthread", 6);
  }
};
Booster booster = new Booster(param, "model.bin");
```

## Prediction
after training and loading a model, you use it to predict other data, the predict results will be a two-dimension float array (nsample, nclass), for predict leaf, it would be (nsample, nclass*ntrees)
```java
DMatrix dtest = new DMatrix("test.svm.txt");
//predict
float[][] predicts = booster.predict(dtest);
//predict leaf
float[][] leafPredicts = booster.predict(dtest, 0, true);
```
