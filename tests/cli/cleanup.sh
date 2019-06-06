rm -f *.model*
pkill xgboost
pkill python
cd ../../rabit/test
make clean;make
cd ../../
make clean;make -j 4
cd tests/cli
