#!/bin/bash
if [ -f MQ2008.rar ]
then
    echo "Use downloaded data to run experiment."
else
    echo "Downloading data."
    wget https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.rar
    unrar x MQ2008.rar
    mv -f MQ2008/Fold1/*.txt .
fi

python trans_data.py train.txt mq2008.train mq2008.train.group

python trans_data.py test.txt mq2008.test mq2008.test.group

python trans_data.py vali.txt mq2008.vali mq2008.vali.group
