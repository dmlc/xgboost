rm -f *.model*
pkill xgboost
pkill python
cd ../../rabit
make clean;make
cd test
make clean;make
cd ../../
make clean;make -j 4
cd tests/cli
