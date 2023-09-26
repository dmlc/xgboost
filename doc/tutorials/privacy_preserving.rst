#############################################
Privacy Preserving Inference with Concrete ML
#############################################

`Concrete ML`_ is a specialized library developed by Zama that allows the execution of machine learning models on encrypted data through `Fully Homomorphic Encryption (FHE) <https://www.youtube.com/watch?v=FFox2S4uqEo>`_, thereby preserving data privacy.

To use models such as XGBClassifier, use the following import:

.. code:: python

  from concrete.ml.sklearn import XGBClassifier

***************************************
Performing Privacy Preserving Inference
***************************************

Initialization of a XGBClassifier can be done as follows:

.. code:: python

  classifier = XGBClassifier(n_bits=6, [other_hyperparameters])


where ``n_bits`` determines the precision of the input features. Note that a higher value of ``n_bits`` increases the precision of the input features and possibly the final model accuracy but also ends up with longer FHE execution time.

Other hyper-parameters that exist in xgboost library can be used.

******************************
Model Training and Compilation
******************************

As commonly used in scikit-learn like models, it can be trained with the .fit() method.

.. code:: python

  classifier.fit(X_train, y_train)

After training, the model can be compiled with a calibration dataset, potentially a subset of the training data:

.. code:: python

  classifier.compile(X_calibrate)

This calibration dataset, ``X_calibrate``, is used in Concrete ML compute the precision (bit-width) of each intermediate value in the model. This is a necessary step to optimize the equivalent FHE circuit.

****************************
FHE Simulation and Execution
****************************

To verify model accuracy in encrypted computations, you can run an FHE simulation:

.. code:: python

  predictions = classifier.predict(X_test, fhe="simulate")

This simulation can be used to evaluate the model. The resulting accuracy of this simulation step is representative of the actual FHE execution without having to pay the cost of an actual FHE execution. 

When the model is ready, actual Fully Homomorphic Encryption execution can be performed:

.. code:: python

  predictions = classifier.predict(X_test, fhe="execute")


Note that using FHE="execute" is a convenient way to assess the model in FHE, but for real deployment, functions to encrypt (on the client), run in FHE (on the server), and finally decrypt (on the client) have to be used for end-to-end privacy-preserving inferences.

Concrete ML provides a deployment API to facilitate this process, ensuring end-to-end privacy.

To go further in the deployment API you can read:

- the `deployment documentation <https://docs.zama.ai/concrete-ml/advanced-topics/client_server>`_
- the `deployment notebook <https://github.com/zama-ai/concrete-ml/blob/17779ca571d20b001caff5792eb11e76fe2c19ba/docs/advanced_examples/ClientServer.ipynb>`_

*******************************
Parameter Tuning in Concrete ML
*******************************

Concrete ML is compatible with standard scikit-learn pipelines such as GridSearchCV or any other hyper-parameter tuning techniques.

******************
Examples and Demos
******************

- `Sentiment analysis (based on transformers + xgboost) <https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis>`_
- `XGBoost Classifier <https://github.com/zama-ai/concrete-ml/blob/6966c84b9698d5418209b346900f81d1270c64bd/docs/advanced_examples/XGBClassifier.ipynb>`_
- `XGBoost Regressor <https://github.com/zama-ai/concrete-ml/blob/6966c84b9698d5418209b346900f81d1270c64bd/docs/advanced_examples/XGBRegressor.ipynb>`_

**********
Conclusion
**********

Concrete ML provides a framework for executing privacy-preserving inferences by leveraging Fully Homomorphic Encryption, allowing secure and private computations on encrypted data.

More information and examples are given in the `Concrete ML documentation`_.

.. _Concrete ML: https://github.com/zama-ai/concrete-ml
.. _`Concrete ML documentation`: https://docs.zama.ai/concrete-ml