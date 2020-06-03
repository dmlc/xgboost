###########################
Swift Package Introduction
###########################
This document gives a basic walkthrough of xgboost swift package.


**List of other helpful links**

* `Swift examples <https://github.com/kongzii/SwiftXGBoost/tree/master/Examples>`_
* `Swift API Reference <https://kongzii.github.io/SwiftXGBoost/>`_


Install XGBoost
---------------
To install XGBoost, please follow instructions at `GitHub <https://github.com/kongzii/SwiftXGBoost#installation>`_.


Include in your project
-----------------------
`SwiftXGBoost <https://github.com/kongzii/SwiftXGBoost>`_ uses `Swift Package Manager <https://swift.org/package-manager/>`_, to use it in your project, simply add it as a dependency in your Package.swift file:

  .. code-block:: swift

    .package(url: "https://github.com/kongzii/SwiftXGBoost.git", from: "0.7.0")


Python compatibility
--------------------
With `PythonKit <https://github.com/pvieito/PythonKit>`_ package, you can import Python modules:

.. code-block:: swift

    let numpy = Python.import("numpy")
    let pandas = Python.import("pandas")

And use them in the same way as in Python:

.. code-block:: swift

    let dataFrame = pandas.read_csv("Examples/Data/veterans_lung_cancer.csv")

And then use them with `SwiftXGBoost <https://github.com/kongzii/SwiftXGBoost>`_, 
check `AftSurvival <https://github.com/kongzii/SwiftXGBoost/blob/master/Examples/AftSurvival/main.swift>`_ for a complete example.


TensorFlow compatibility
------------------------
If you are using `S4TF toolchains <https://github.com/tensorflow/swift>`_, you can utilize tensors directly:

.. code-block:: swift

    let tensor = Tensor<Float>(shape: TensorShape([2, 3]), scalars: [1, 2, 3, 4, 5, 6])
    let tensorData = try DMatrix(name: "tensorData", from: tensor)


Data Interface
--------------
The XGBoost swift package is currently able to load data from:

- LibSVM text format file
- Comma-separated values (CSV) file
- NumPy 2D array
- `Swift for Tensorflow <https://www.tensorflow.org/swift/>`_  2D Tensor
- XGBoost binary buffer file

The data is stored in a `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_ class.

To load a libsvm text file into `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_ class:

  .. code-block:: swift

    let svmData = try DMatrix(name: "train", from: "Examples/Data/data.svm.txt", format: .libsvm)

To load a CSV file into `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_:

  .. code-block:: swift

    # labelColumn specifies the index of the column containing the true label
    let csvData = try DMatrix(name: "train", from: "Examples/Data/data.csv", format: .csv, labelColumn: 0)

  .. note:: Use Pandas to load CSV files with headers.

    Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas (see below) to read CSV files with headers.

To load a NumPy array into `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_:

  .. code-block:: swift

    let numpyData = try DMatrix(name: "train", from: numpy.random.rand(5, 10), label: numpy.random.randint(2, size: 5))

To load a Pandas data frame into `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_:

  .. code-block:: swift

    let pandasDataFrame = pandas.DataFrame(numpy.arange(12).reshape([4, 3]), columns: ["a", "b", "c"])
    let pandasLabel = numpy.random.randint(2, size: 4)
    let pandasData = try DMatrix(name: "data", from: pandasDataFrame.values, label: pandasLabel)

Saving `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_ into an XGBoost binary file will make loading faster:

  .. code-block:: swift

    try pandasData.save(to: "train.buffer")

Missing values can be replaced by a default value in the `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_ constructor:

  .. code-block:: swift

    let dataWithMissingValues = try DMatrix(name: "data", from: pandasDataFrame.values, missingValue: 999.0)

Various `float fields <https://kongzii.github.io/SwiftXGBoost/Enums/FloatField.html>`_  and `uint fields <https://kongzii.github.io/SwiftXGBoost/Enums/UIntField.html>`_ can be set when needed:

  .. code-block:: swift

    try dataWithMissingValues.set(field: .weight, values: [Float](repeating: 1, count: try dataWithMissingValues.rowCount()))

And returned:

  .. code-block:: swift

    let labelsFromData = try pandasData.get(field: .label)


Setting Parameters
------------------
Parameters for `Booster <https://kongzii.github.io/SwiftXGBoost/Classes/Booster.html>`_ can also be set.

Using the set method:

  .. code-block:: swift

    let firstBooster = try Booster()
    try firstBooster.set(parameter: "tree_method", value: "hist")

Or as a list at initialization:

  .. code-block:: swift

    let parameters = [Parameter(name: "tree_method", value: "hist")]
    let secondBooster = try Booster(parameters: parameters)


Training
--------
Training a model requires a booster and a dataset.

.. code-block:: swift

  let trainingData = try DMatrix(name: "train", from: "Examples/Data/data.csv", format: .csv, labelColumn: 0)
  let boosterWithCachedData = try Booster(with: [trainingData])
  try boosterWithCachedData.train(iterations: 100, trainingData: trainingData)

After training, the model can be saved:

.. code-block:: swift

  try boosterWithCachedData.save(to: "0001.xgboost")

The model can also be dumped to a text:

.. code-block:: swift

  let textModel = try boosterWithCachedData.dumped(format: .text)

A saved model can be loaded as follows:

.. code-block:: swift

  let loadedBooster = try Booster(from: "0001.xgboost")


Prediction
----------
A model that has been trained or loaded can perform predictions on data sets.

From Numpy array:

.. code-block:: swift

  let testDataNumpy = try DMatrix(name: "test", from: numpy.random.rand(7, 12))
  let predictionNumpy = try loadedBooster.predict(from: testDataNumpy)

From Swift array:

.. code-block:: swift

  let testData = try DMatrix(name: "test", from: [69.0,60.0,7.0,0,0,0,1,1,0,1,0,0], shape: Shape(1, 12))
  let prediction = try loadedBooster.predict(from: testData)


Plotting
--------
You can also save the plot of importance into a file:

.. code-block:: swift

  try boosterWithCachedData.saveImportanceGraph(to: "importance") // .png extension will be added


C API
--------
Both `Booster <https://kongzii.github.io/SwiftXGBoost/Classes/Booster.html>`_ and `DMatrix <https://kongzii.github.io/SwiftXGBoost/Classes/DMatrix.html>`_ are exposing pointers to the underlying C.

You can import a C-API library:

.. code-block:: swift

  import CXGBoost

And use it directly in your Swift code:

.. code-block:: swift

  try safe { XGBoosterSaveModel(boosterWithCachedData.booster, "0002.xgboost") }

`safe` is a helper function that will throw an error if C-API call fails.


More
--------
For more details and examples, check out `GitHub repository <https://github.com/kongzii/SwiftXGBoost>`_.
