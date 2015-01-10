#!/bin/bash

if [ "$#" -lt 2 ];
then
    echo "Usage: <nworkers> <path_in_HDFS>"
    exit -1
fi

curDir=`pwd`
dataDir=../../demo/binary_classification
trainFile=$dataDir/agaricus.txt.train
input=$2
output=$2/model

# generate the training file if it doesnot exist
if [ ! -f "$trainFile" ];
then 
  echo "Generating training file:"
  cd $dataDir
  # map feature using indicator encoding, also produce featmap.txt
  python mapfeat.py
  # split train and test
  python mknfold.py agaricus.txt 1
  cd $curDir
fi

hadoop fs -mkdir $input
hadoop fs -put $trainFile $input
#hadoop fs -rm -skipTrash -r $output

# training and output the final model file
python ../../rabit/tracker/rabit_hadoop.py -n $1 -i $input/agaricus.txt.train -o $output -f $dataDir/mushroom.hadoop.conf \
    --jobname xgboost_hadoop ../../xgboost mushroom.hadoop.conf data=stdin model_out=stdout

# get the final model file
hadoop fs -get $output/part-00000 ./final.model
# output prediction task=pred 
../../xgboost $dataDir/mushroom.hadoop.conf task=pred model_in=final.model
# print the boosters of 00002.model in dump.raw.txt
../../xgboost $dataDir/mushroom.hadoop.conf task=dump model_in=final.model name_dump=dump.raw.txt
# use the feature map in printing for better visualization
../../xgboost $dataDir/mushroom.hadoop.conf task=dump model_in=final.model fmap=$dataDir/featmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
