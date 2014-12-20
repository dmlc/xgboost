export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -Lrabit/lib -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC  -Irabit/src

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# by default use c++11
ifeq ($(no_cxx11),1)
else 
	CFLAGS += -std=c++11
endif

# specify tensor path
BIN = xgboost
OBJ = updater.o gbm.o io.o main.o 
MPIBIN = xgboost-mpi
SLIB = wrapper/libxgboostwrapper.so 

.PHONY: clean all mpi python Rpack librabit librabit_mpi

all: $(BIN) $(OBJ) $(SLIB) mpi
mpi: $(MPIBIN)

# rules to get rabit library
librabit:
	if [ ! -d rabit ]; then git clone https://github.com/tqchen/rabit.git; fi
	cd rabit;make lib/librabit.a; cd -
librabit_mpi:
	if [ ! -d rabit ]; then git clone https://github.com/tqchen/rabit.git; fi
	cd rabit;make lib/librabit_mpi.a; cd -

python: wrapper/libxgboostwrapper.so
# now the wrapper takes in two files. io and wrapper part
updater.o: src/tree/updater.cpp  src/tree/*.hpp src/*.h src/tree/*.h src/utils/*.h
gbm.o: src/gbm/gbm.cpp src/gbm/*.hpp src/gbm/*.h 
io.o: src/io/io.cpp src/io/*.hpp src/utils/*.h src/learner/dmatrix.h src/*.h
main.o: src/xgboost_main.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h 
xgboost-mpi:  updater.o gbm.o io.o main.o librabit_mpi
xgboost:  updater.o gbm.o io.o main.o  librabit
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h  updater.o gbm.o io.o librabit

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)  -lrabit

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)  -lrabit

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) ) 

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS) -lrabit_mpi

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

Rpack:
	make clean
	rm -rf xgboost xgboost*.tar.gz
	cp -r R-package xgboost
	rm -rf xgboost/inst/examples/*.buffer
	rm -rf xgboost/inst/examples/*.model
	rm -rf xgboost/inst/examples/dump*
	rm -rf xgboost/src/*.o xgboost/src/*.so xgboost/src/*.dll
	rm -rf xgboost/demo/*.model xgboost/demo/*.buffer xgboost/demo/*.txt
	rm -rf xgboost/demo/runall.R
	cp -r src xgboost/src/src
	mkdir xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.h xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.cpp xgboost/src/wrapper
	cp ./LICENSE xgboost
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars
	cat R-package/src/Makevars.win|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars.win
	R CMD build xgboost
	rm -rf xgboost
	R CMD check --as-cran xgboost*.tar.gz

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
