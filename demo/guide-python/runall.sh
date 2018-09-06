#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../../python-package
python basic_walkthrough.py
python custom_objective.py
python boost_from_prediction.py
python predict_first_ntree.py
python generalized_linear_model.py
python cross_validation.py
python predict_leaf_indices.py
python sklearn_examples.py
python sklearn_parallel.py
python external_memory.py
rm -rf *~ *.model *.buffer
