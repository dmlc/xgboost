xgboost4j : java wrapper for xgboost
====

This page will introduce xgboost4j, the java wrapper for xgboost, including:
* [Building](#build-xgboost4j)
* [Data Interface](#data-interface)
* [Setting Parameters](#setting-parameters)
* [Train Model](#training-model)
* [Prediction](#prediction)

=
#### Build xgboost4j
* Build native library  
first make sure you have installed jdk and `JAVA_HOME` has been setted properly, then simply run `./create_wrap.sh`.

* Package xgboost4j  
to package xgboost4j, you can run `mvn package` in xgboost4j folder or just use IDE(eclipse/netbeans) to open this maven project and build.

=
#### Data Interface
Like the xgboost python module, xgboost4j use ```DMatrix``` to handle data, libsvm txt format file, sparse matrix in CSR/CSC format, and dense matrix is supported.

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

#### Setting Parameters
* A util class ```Params``` in xgboost4j is used to handle parameters.
* To import ```Params``` :
```java
import org.dmlc.xgboost4j.util.Params;
```
* to set parameters :
```java
Params params = new Params() {
  {
    put("eta", "1.0");
    put("max_depth", "2");
    put("silent", "1");
    put("objective", "binary:logistic");
    put("eval_metric", "logloss");
  }
};
```
* Multiple values with same param key is handled naturally in ```Params```, e.g. :
```java
Params params = new Params() {
  {
    put("eta", "1.0");
    put("max_depth", "2");
    put("silent", "1");
    put("objective", "binary:logistic");
    put("eval_metric", "logloss");
    put("eval_metric", "error");
  }
};
```

#### Training Model
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
DMatrix[] evalMats = new DMatrix[] {trainMat, validMat};
String[] evalNames = new String[] {"train", "valid"};
int round = 2;
Booster booster = Trainer.train(params, trainMat, round, evalMats, evalNames, null, null);
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
    put("silent", "1");
    put("nthread", "6");
  }
};
Booster booster = new Booster(param, "model.bin");
```

####Prediction
after training and loading a model, you use it to predict other data, the predict results will be a two-dimension float array (nsample, nclass) ,for predict leaf, it would be (nsample, nclass*ntrees)
```java
DMatrix dtest = new DMatrix("test.svm.txt");
//predict
float[][] predicts = booster.predict(dtest);
//predict leaf
float[][] leafPredicts = booster.predict(dtest, 0, true);
```
