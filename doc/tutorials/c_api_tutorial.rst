##############
C API Tutorial
##############

In this tutorial, we are going to install XGBoost library & configure the CMakeLists.txt file of our C/C++ application to link XGBoost library with our application. Later on, we will see some useful tips for using C API and code snippets as examples to use various functions available in C API to perform basic task like loading, training model & predicting on test dataset.

.. contents::
  :backlinks: none
  :local:

************
Requirements
************

Install CMake - Follow the `cmake installation documentation <https://cmake.org/install/>`_ for instructions.
Install Conda - Follow the `conda installation  documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for instructions

*************************************
Install XGBoost on conda environment
*************************************

Run the following commands on your terminal. The below commands will install the XGBoost in your XGBoost folder of the repository cloned

.. code-block:: bash

    # clone the XGBoost repository & its submodules
    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost
    mkdir build
    cd build
    # Activate the Conda environment, into which we'll install XGBoost
    conda activate [env_name]
    # Build the compiled version of XGBoost inside the build folder
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    # install XGBoost in your conda environment (usually under [your home directory]/miniconda3)
    make install

*********************************************************************
Configure CMakeList.txt file of your application to link with XGBoost
*********************************************************************

Here, we assume that your C++ application is using CMake for builds.

Use ``find_package()`` and ``target_link_libraries()`` in your application's CMakeList.txt to link with the XGBoost library:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.13)
    project(your_project_name LANGUAGES C CXX VERSION your_project_version)
    find_package(xgboost REQUIRED)
    add_executable(your_project_name /path/to/project_file.c)
    target_link_libraries(your_project_name xgboost::xgboost)

To ensure that CMake can locate the XGBoost library, supply ``-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`` argument when invoking CMake. This option instructs CMake to locate the XGBoost library in ``$CONDA_PREFIX``, which is where your Conda environment is located.

.. code-block:: bash

  # Nagivate to the build directory for your application
  cd build
  # Activate the Conda environment where we previously installed XGBoost
  conda activate [env_name]
  # Invoke CMake with CMAKE_PREFIX_PATH
  cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
  # Build your application
  make

************************
Usefull Tips To Remember
************************

Below are some useful tips while using C API:

1. Error handling: Always check the return value of the C API functions.

a. In a C application: Use the following macro to guard all calls to XGBoost's C API functions. The macro prints all the error/ exception occurred:

.. highlight:: c
   :linenothreshold: 5

.. code-block:: c

  #define safe_xgboost(call) {  \
    int err = (call); \
    if (err != 0) { \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError());  \
      exit(1); \
    } \
  }

In your application, wrap all C API function calls with the macro as follows:

.. code-block:: c

  DMatrixHandle train;
  safe_xgboost(XGDMatrixCreateFromFile("/path/to/training/dataset/", silent, &train));

b. In a C++ application: modify the macro ``safe_xgboost`` to throw an exception upon an error.

.. highlight:: cpp
   :linenothreshold: 5

.. code-block:: cpp

  #define safe_xgboost(call) {  \
    int err = (call); \
    if (err != 0) { \
      throw new Exception(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                          ": error in " + #call + ":" + XGBGetLastError()));  \
    } \
  }

c. Assertion technique: It works both in C/ C++. If expression evaluates to 0 (false), then the expression, source code filename, and line number are sent to the standard error, and then abort() function is called. It can be used to test assumptions made by you in the code.

.. code-block:: c

  DMatrixHandle dmat;
  assert( XGDMatrixCreateFromFile("training_data.libsvm", 0, &dmat) == 0);


2. Always remember to free the allocated space by BoosterHandle & DMatrixHandle appropriately:

.. code-block:: c

    #include <assert.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <xgboost/c_api.h>

    int main(int argc, char** argv) {
      int silent = 0;

      BoosterHandle booster;

      // do something with booster

      //free the memory
      XGBoosterFree(booster)

      DMatrixHandle DMatrixHandle_param;

      // do something with DMatrixHandle_param

      // free the memory
      XGDMatrixFree(DMatrixHandle_param);

      return 0;
    }


3. For tree models, it is important to use consistent data formats during training and scoring/ predicting otherwise it will result in wrong outputs.
   Example if we our training data is in ``dense matrix`` format then your prediction dataset should also be a ``dense matrix`` or if training in ``libsvm`` format then dataset for prediction should also be in ``libsvm`` format.


4. Always use strings for setting values to the parameters in booster handle object. The paramter value can be of any data type (e.g. int, char, float, double, etc), but they should always be encoded as strings.

.. code-block:: c

    BoosterHandle booster;
    XGBoosterSetParam(booster, "paramter_name", "0.1");


**************************************************************
Sample examples along with Code snippet to use C API functions
**************************************************************

1. If the dataset is available in a file, it can be loaded into a ``DMatrix`` object using the `XGDMatrixCreateFromFile <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a357c3654a1a4dcc05e6b5c50acd17105>`_

.. code-block:: c

  DMatrixHandle data; // handle to DMatrix
  // Load the dat from file & store it in data variable of DMatrixHandle datatype
  safe_xgboost(XGDMatrixCreateFromFile("/path/to/file/filename", silent, &data));


2. You can also create a ``DMatrix`` object from a 2D Matrix using the `XGDMatrixCreateFromMat function <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a079f830cb972df70c7f50fb91678d62f>`_

.. code-block:: c

  // 1D matrix
  const int data1[] = { 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };

  // 2D matrix
  const int ROWS = 5, COLS = 3;
  const int data2[ROWS][COLS] = { {1, 2, 3}, {2, 4, 6}, {3, -1, 9}, {4, 8, -1}, {2, 5, 1}, {0, 1, 5} };
  DMatrixHandle dmatrix1, dmatrix2;
  // Pass the matrix, no of rows & columns contained in the matrix variable
  // here '0' represents the missing value in the matrix dataset
  // dmatrix variable will contain the created DMatrix using it
  safe_xgboost(XGDMatrixCreateFromMat(data1, 1, 50, 0, &dmatrix));
  // here -1 represents the missing value in the matrix dataset
  safe_xgboost(XGDMatrixCreateFromMat(data2, ROWS, COLS, -1, &dmatrix2));


3. Create a Booster object for training & testing on dataset using `XGBoosterCreate <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#ad9fe6f8c8c4901db1c7581a96a21f9ae>`_

.. code-block:: c

  BoosterHandle booster;
  const int eval_dmats_size;
  // We assume that training and test data have been loaded into 'train' and 'test'
  DMatrixHandle eval_dmats[eval_dmats_size] = {train, test};
  safe_xgboost(XGBoosterCreate(eval_dmats, eval_dmats_size, &booster));


4. For each ``DMatrix`` object, set the labels using `XGDMatrixSetFloatInfo <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#aef75cda93db3ae9af89e465ae7e9cbe3>`_. Later you can access the label using `XGDMatrixGetFloatInfo <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#ab0ee317539a1fb1ce2b5f249e8c768f6>`_.

.. code-block:: c

  const int ROWS=5, COLS=3;
  const int data[ROWS][COLS] = { {1, 2, 3}, {2, 4, 6}, {3, -1, 9}, {4, 8, -1}, {2, 5, 1}, {0, 1, 5} };
  DMatrixHandle dmatrix;

  safe_xgboost(XGDMatrixCreateFromMat(data, ROWS, COLS, -1, &dmatrix));

  // variable to store labels for the dataset created from above matrix
  float labels[ROWS];

  for (int i = 0; i < ROWS; i++) {
    labels[i] = i;
  }

  // Loading the labels
  safe_xgboost(XGDMatrixSetFloatInfo(dmatrix, "label", labels, ROWS));

  // reading the labels and store the length of the result
  bst_ulong result_len;

  // labels result
  const float *result;

  safe_xgboost(XGDMatrixGetFloatInfo(dmatrix, "label", &result_len, &result));

  for(unsigned int i = 0; i < result_len; i++) {
    printf("label[%i] = %f\n", i, result[i]);
  }


5. Set the parameters for the ``Booster`` object according to the requirement using `XGBoosterSetParam <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#af7378865b0c999d2d08a5b16483b8bcb>`_ . Check out the full list of parameters available `here <https://xgboost.readthedocs.io/en/latest/parameter.html>`_ .

.. code-block :: c

    BoosterHandle booster;
    safe_xgboost(XGBoosterSetParam(booster, "booster", "gblinear"));
    // default max_depth =6
    safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
    // default eta  = 0.3
    safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));


6. Train & evaluate the model using `XGBoosterUpdateOneIter <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a13594d68b27327db290ec5e0a0ac92ae>`_ and `XGBoosterEvalOneIter <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a201b53edb9cc52e9def1ccea951d18fe>`_ respectively.

.. code-block:: c

    int num_of_iterations = 20;
    const char* eval_names[eval_dmats_size] = {"train", "test"};
    const char* eval_result = NULL;

    for (int i = 0; i < num_of_iterations; ++i) {
      // Update the model performance for each iteration
      safe_xgboost(XGBoosterUpdateOneIter(booster, i, train));

      // Give the statistics for the learner for training & testing dataset in terms of error after each iteration
      safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, eval_dmats_size, &eval_result));
      printf("%s\n", eval_result);
    }

.. note:: For customized loss function, use `XGBoosterBoostOneIter function <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#afd4a42c38cfb16d2cf2a9cf5daba4e83>`_ instead and manually specify the gradient and 2nd order gradient.


7.  Predict the result on a test set using `XGBoosterPredict <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#adc14afaedd5f1add105d18942a4de33c>`_

.. code-block:: c

    bst_ulong output_length;

    const float *output_result;
    safe_xgboost(XGBoosterPredict(booster, test, 0, 0, &output_length, &output_result));

    for (unsigned int i = 0; i < output_length; i++){
      printf("prediction[%i] = %f \n", i, output_result[i]);
    }


8. Free all the internal structure used in your code using `XGDMatrixFree <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#af06a15433b01e3b8297930a38155e05d>`_ and `XGBoosterFree <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a5d816936b005a103f0deabf287a6a5da>`_. This step is important to prevent memory leak.

.. code-block:: c

  safe_xgboost(XGDMatrixFree(dmatrix));
  safe_xgboost(XGBoosterFree(booster));


9. Get the number of features in your dataset using `XGBoosterGetNumFeature <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#aa2c22f65cf2770c0e2e56cc7929a14af>`_.

.. code-block:: c

    bst_ulong num_of_features = 0;

    // Assuming booster variable of type BoosterHandle is already declared
    // and dataset is loaded and trained on booster
    // storing the results in num_of_features variable
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_of_features));

    // Printing number of features by type conversion of num_of_features variable from bst_ulong to unsigned long
    printf("num_feature: %lu\n", (unsigned long)(num_of_features));


10. Load the model using `XGBoosterLoadModel function <https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a054571e6364f9a1cbf6b6b4fd2f156d6>`_

.. code-block:: c

    BoosterHandle booster;
    const char *model_path = "/path/of/model";

    // create booster handle first
    safe_xgboost(XGBoosterCreate(NULL, 0, &booster));

    // set the model parameters here

    // load model
    safe_xgboost(XGBoosterLoadModel(booster, model_path));

    // predict the model here
