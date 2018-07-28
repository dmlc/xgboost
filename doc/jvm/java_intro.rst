##################
XGBoost4J Java API
##################
This tutorial introduces Java API for XGBoost.

**************
Data Interface
**************
Like the XGBoost python module, XGBoost4J uses ``DMatrix`` to handle data,
libsvm txt format file, sparse matrix in CSR/CSC format, and dense matrix is
supported.

* The first step is to import ``DMatrix``:

  .. code-block:: java

    import org.dmlc.xgboost4j.DMatrix;

* Use ``DMatrix`` constructor to load data from a libsvm text format file:

  .. code-block:: java

    DMatrix dmat = new DMatrix("train.svm.txt");

* Pass arrays to ``DMatrix`` constructor to load from sparse matrix.

  Suppose we have a sparse matrix
  
  .. code-block:: none
  
    1 0 2 0
    4 0 0 3
    3 1 2 0
  
  We can express the sparse matrix in `Compressed Sparse Row (CSR) <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_ format:
  
  .. code-block:: java
  
    long[] rowHeaders = new long[] {0,2,4,7};
    float[] data = new float[] {1f,2f,4f,3f,3f,1f,2f};
    int[] colIndex = new int[] {0,2,0,3,0,1,2};
    DMatrix dmat = new DMatrix(rowHeaders, colIndex, data, DMatrix.SparseType.CSR);
  
  ... or in `Compressed Sparse Column (CSC) <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>`_ format:
  
  .. code-block:: java
  
    long[] colHeaders = new long[] {0,3,4,6,7};
    float[] data = new float[] {1f,4f,3f,1f,2f,2f,3f};
    int[] rowIndex = new int[] {0,1,2,2,0,2,1};
    DMatrix dmat = new DMatrix(colHeaders, rowIndex, data, DMatrix.SparseType.CSC);

* You may also load your data from a dense matrix. Let's assume we have a matrix of form

  .. code-block:: none

    1    2
    3    4
    5    6

  Using `row-major layout <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_, we specify the dense matrix as follows:

  .. code-block:: java

    float[] data = new float[] {1f,2f,3f,4f,5f,6f};
    int nrow = 3;
    int ncol = 2;
    float missing = 0.0f;
    DMatrix dmat = new Matrix(data, nrow, ncol, missing);

* To set weight:

  .. code-block:: java

    float[] weights = new float[] {1f,2f,1f};
    dmat.setWeight(weights);

******************
Setting Parameters
******************
* In XGBoost4J any ``Iterable<Entry<String, Object>>`` object could be used as parameters.

* To set parameters, for non-multiple value params, you can simply use entrySet of an Map:

  .. code-block:: java

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

* for the situation that multiple values with same param key, List<Entry<String, Object>> would be a good choice, e.g. :

  .. code-block:: java

    List<Entry<String, Object>> params = new ArrayList<Entry<String, Object>>() {
        {
            add(new SimpleEntry<String, Object>("eta", 1.0));
            add(new SimpleEntry<String, Object>("max_depth", 2.0));
            add(new SimpleEntry<String, Object>("silent", 1));
            add(new SimpleEntry<String, Object>("objective", "binary:logistic"));
        }
    };

**************
Training Model
**************
With parameters and data, you are able to train a booster model.

* Import ``Trainer`` and ``Booster``:

  .. code-block:: java

    import org.dmlc.xgboost4j.Booster;
    import org.dmlc.xgboost4j.util.Trainer;

* Training

  .. code-block:: java

    DMatrix trainMat = new DMatrix("train.svm.txt");
    DMatrix validMat = new DMatrix("valid.svm.txt");
    //specify a watchList to see the performance
    //any Iterable<Entry<String, DMatrix>> object could be used as watchList
    List<Entry<String, DMatrix>> watchs =  new ArrayList<>();
    watchs.add(new SimpleEntry<>("train", trainMat));
    watchs.add(new SimpleEntry<>("test", testMat));
    int round = 2;
    Booster booster = Trainer.train(params, trainMat, round, watchs, null, null);

* Saving model

  After training, you can save model and dump it out.

  .. code-block:: java

    booster.saveModel("model.bin");

* Dump Model and Feature Map

  .. code-block:: java

    booster.dumpModel("modelInfo.txt", false)
    //dump with featureMap
    booster.dumpModel("modelInfo.txt", "featureMap.txt", false)

* Load a model

  .. code-block:: java

    Params param = new Params() {
      {
        put("silent", 1);
        put("nthread", 6);
      }
    };
    Booster booster = new Booster(param, "model.bin");

**********
Prediction
**********
After training and loading a model, you can use it to make prediction for other data. The result will be a two-dimension float array ``(nsample, nclass)``; for ``predictLeaf()``, the result would be of shape ``(nsample, nclass*ntrees)``.

.. code-block:: java

  DMatrix dtest = new DMatrix("test.svm.txt");
  //predict
  float[][] predicts = booster.predict(dtest);
  //predict leaf
  float[][] leafPredicts = booster.predict(dtest, 0, true);

