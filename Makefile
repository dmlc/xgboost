export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# by default use c++11
ifeq ($(cxx11),1)
	CFLAGS += -std=c++11
else 
endif

# specify tensor path
BIN = xgboost 
MOCKBIN = xgboost.mock
OBJ = updater.o gbm.o io.o main.o 
MPIBIN = xgboost.mpi
SLIB = wrapper/libxgboostwrapper.so 

.PHONY: clean all mpi python Rpack

all: $(BIN) $(OBJ) $(SLIB) $(MOCKBIN)
mpi: $(MPIBIN)

python: wrapper/libxgboostwrapper.so
# now the wrapper takes in two files. io and wrapper part
updater.o: src/tree/updater.cpp  src/tree/*.hpp src/*.h src/tree/*.h src/utils/*.h
gbm.o: src/gbm/gbm.cpp src/gbm/*.hpp src/gbm/*.h 
io.o: src/io/io.cpp src/io/*.hpp src/utils/*.h src/learner/dmatrix.h src/*.h
main.o: src/xgboost_main.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h 
xgboost.mpi:  updater.o gbm.o io.o main.o subtree/rabit/lib/librabit_mpi.a
xgboost.mock: updater.o gbm.o io.o main.o subtree/rabit/lib/librabit_mock.a
xgboost:  updater.o gbm.o io.o main.o subtree/rabit/lib/librabit.a
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h  updater.o gbm.o io.o subtree/rabit/lib/librabit.a

# dependency on rabit
subtree/rabit/lib/librabit.a: subtree/rabit/src/engine.cc
	cd subtree/rabit;make lib/librabit.a; cd ../..
subtree/rabit/lib/librabit_empty.a: subtree/rabit/src/engine_empty.cc
	cd subtree/rabit;make lib/librabit_empty.a; cd ../..
subtree/rabit/lib/librabit_mock.a: subtree/rabit/src/engine_mock.cc
	cd subtree/rabit;make lib/librabit_mock.a; cd ../..
subtree/rabit/lib/librabit_mpi.a: subtree/rabit/src/engine_mpi.cc
	cd subtree/rabit;make lib/librabit_mpi.a; cd ../..

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS) 

$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS) 

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS) 

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) ) 

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS) 

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

Rpack:
	make clean
	cd subtree/rabit;make clean;cd ..
	rm -rf xgboost xgboost*.tar.gz
	cp -r R-package xgboost
	rm -rf xgboost/inst/examples/*.buffer
	rm -rf xgboost/inst/examples/*.model
	rm -rf xgboost/inst/examples/dump*
	rm -rf xgboost/src/*.o xgboost/src/*.so xgboost/src/*.dll
	rm -rf subtree/rabit/src/*.o
	rm -rf xgboost/demo/*.model xgboost/demo/*.buffer xgboost/demo/*.txt
	rm -rf xgboost/demo/runall.R
	cp -r src xgboost/src/src
	cp -r subtree xgboost/src/subtree
	mkdir xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.h xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.cpp xgboost/src/wrapper
	cp ./LICENSE xgboost
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars
	cp xgboost/src/Makevars xgboost/src/Makevars.win
	R CMD build xgboost
	rm -rf xgboost
	R CMD check --as-cran xgboost*.tar.gz

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
	cd subtree/rabit; make clean; cd ..
