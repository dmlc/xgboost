Highlights
=====
Higgs challenge ends recently, xgboost is being used by many users. This list highlights the xgboost solutions of players
* Blogpost by phunther: [Winning solution of Kaggle Higgs competition: what a single model can do](http://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/)
* The solution by Tianqi Chen and Tong He [Link](https://github.com/hetong007/higgsml)

Guide for Kaggle Higgs Challenge
=====

This is the folder giving example of how to use XGBoost Python Module  to run Kaggle Higgs competition

This script will achieve about 3.600 AMS score in public leaderboard. To get start, you need do following step:

1. Compile the XGBoost python lib
```bash
cd ../..
make
```

2. Put training.csv test.csv on folder './data' (you can create a symbolic link)

3. Run ./run.sh

Speed
=====
speedtest.py compares xgboost's speed on this dataset with sklearn.GBM


Using R module
=====
* Alternatively, you can run using R, higgs-train.R and higgs-pred.R.
