#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py agaricus.txt 1
# training
../../xgboost mushroom.conf num_round=1 model_out=full.model bst:max_depth=3
../../xgboost mushroom.conf task=dump model_in=full.model fmap=featmap.txt name_dump=dump.full.txt

# constrain
../../xgboost mushroom.conf num_round=1 model_out=ban.model bst:max_depth=3 bst:fban=22-31 

# constrain
../../xgboost mushroom.conf num_round=1 model_out=pass.model bst:max_depth=3 bst:fdefault=-1 bst:fpass=22-31

../../xgboost mushroom.conf task=dump model_in=ban.model fmap=featmap.txt name_dump=dump.ban.txt
../../xgboost mushroom.conf task=dump model_in=pass.model fmap=featmap.txt name_dump=dump.pass.txt

echo "========full======="
cat dump.full.txt

echo "========ban======="
cat dump.ban.txt

echo "========pass======="
cat dump.pass.txt
