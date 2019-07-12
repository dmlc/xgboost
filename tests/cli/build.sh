rm -f *.model*
cd ../../rabit
make clean;make
cd test
make clean;make
cd ../../
make clean;make -j 8
cd tests/cli
