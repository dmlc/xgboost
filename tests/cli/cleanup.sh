rm -f *.model*
pkill xgboost
pkill python
cd ../../
make clean;make -j 8
cd tests/cli
