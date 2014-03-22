#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py agaricus.txt 1
# training
../../xgboost mushroom.conf num_round=2 model_out=full.model bst:max_depth=3
../../xgboost mushroom.conf task=dump model_in=full.model fmap=featmap.txt name_dump=dump.full.txt

# major element of batch running: add batch prefix to each setting, batch:run=1 will run that action

../../xgboost mushroom.conf model_in=full.model model_out=m1.model task=interact\
 batch:interact:booster_index=0 batch:bst:interact:remove=1 batch:run=1\
 batch:interact:booster_index=1 batch:bst:interact:remove=1 batch:run=1\
 batch:interact:booster_index=1 batch:bst:interact:expand=9 batch:run=1\

../../xgboost mushroom.conf task=dump model_in=m1.model fmap=featmap.txt name_dump=dump.m1.txt

echo "========full======="
cat dump.full.txt

echo "========m1======="
cat dump.m1.txt
