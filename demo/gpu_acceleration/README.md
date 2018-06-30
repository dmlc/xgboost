# GPU Acceleration Demo

This demo shows how to train a model on the [forest cover type](https://archive.ics.uci.edu/ml/datasets/covertype) dataset using GPU acceleration. The forest cover type dataset has 581,012 rows and 54 features, making it time consuming to process. We compare the run-time and accuracy of the GPU and CPU histogram algorithms.

This demo requires the [GPU plug-in](https://xgboost.readthedocs.io/en/latest/gpu/index.html) to be built and installed.

The dataset is automatically loaded via the sklearn script. 
