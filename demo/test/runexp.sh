#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py agaricus.txt 1
# training
../../xgboost mushroom.conf num_round=1 model_out=full.model bst:max_depth=3
../../xgboost mushroom.conf task=dump model_in=full.model fmap=featmap.txt name_dump=dump.full.txt

# training
../../xgboost mushroom.conf num_round=2 model_out=m1.model bst:max_depth=1

# this is what dump will looklike with feature map
../../xgboost mushroom.conf task=dump model_in=m1.model fmap=featmap.txt name_dump=dump.m1.txt

# interaction
../../xgboost mushroom.conf task=interact model_in=m1.model model_out=m2.model interact:booster_index=0 bst:interact:expand=1
../../xgboost mushroom.conf task=interact model_in=m2.model model_out=m3.model interact:booster_index=0 interact:action=remove


# this is what dump will looklike with feature map
../../xgboost mushroom.conf task=dump model_in=m1.model fmap=featmap.txt name_dump=dump.m2.txt
../../xgboost mushroom.conf task=dump model_in=m2.model fmap=featmap.txt name_dump=dump.m2.txt
../../xgboost mushroom.conf task=dump model_in=m3.model fmap=featmap.txt name_dump=dump.m3.txt

echo "========m1======="
cat dump.m1.txt

echo "========m2========"
cat dump.m2.txt

echo "========m3========"
cat dump.m3.txt



echo "========full======="
cat dump.full.txt
