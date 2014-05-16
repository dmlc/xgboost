#Download the dataset from web site
wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar

#please first install the unrar package
unrar x MQ2008

python MQ2008/Fold1/trans_data.py train.txt mq2008.train mq2008.train.group

python MQ2008/Fold1/trans_data.py test.txt mq2008.test mq2008.test.group

python MQ2008/Fold1/trans_data.py vali.txt mq2008.vali mq2008.vali.group

../../xgboost mq2008.conf

../../xgboost mq2008.conf task=pred model_in=0002.model

../../xgboost mq2008.conf task=dump model_in=0002.model name_dump=dump.raw.txt
