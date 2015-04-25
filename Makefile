export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC

ifeq ($(OS), Windows_NT)
	export CXX = g++ -m64
	export CC = gcc -m64
endif

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

# handling dmlc
ifdef dmlc
	ifndef config
		ifneq ("$(wildcard $(dmlc)/config.mk)","")
			config = $(dmlc)/config.mk
		else
			config = $(dmlc)/make/config.mk
		endif	
	endif
	include $(config)
	include $(dmlc)/make/dmlc.mk
	LDFLAGS+= $(DMLC_LDFLAGS)
	LIBDMLC=$(dmlc)/libdmlc.a
else
	LIBDMLC=dmlc_simple.o
endif

ifeq ($(OS), Windows_NT)
	LIBRABIT = subtree/rabit/lib/librabit_empty.a
	SLIB = wrapper/xgboost_wrapper.dll
else
	LIBRABIT = subtree/rabit/lib/librabit.a
	SLIB = wrapper/libxgboostwrapper.so
endif

# specify tensor path
BIN = xgboost
MOCKBIN = xgboost.mock
OBJ = updater.o gbm.o io.o main.o dmlc_simple.o
MPIBIN =
TARGET = $(BIN) $(OBJ) $(SLIB)

.PHONY: clean all mpi python Rpack

all: $(BIN) $(OBJ) $(SLIB)
mpi: $(MPIBIN)

python: wrapper/libxgboostwrapper.so
# now the wrapper takes in two files. io and wrapper part
updater.o: src/tree/updater.cpp  src/tree/*.hpp src/*.h src/tree/*.h src/utils/*.h
dmlc_simple.o: src/io/dmlc_simple.cpp src/utils/*.h
gbm.o: src/gbm/gbm.cpp src/gbm/*.hpp src/gbm/*.h 
io.o: src/io/io.cpp src/io/*.hpp src/utils/*.h src/learner/dmatrix.h src/*.h
main.o: src/xgboost_main.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h 
xgboost:  updater.o gbm.o io.o main.o $(LIBRABIT) $(LIBDMLC)
wrapper/xgboost_wrapper.dll wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h  updater.o gbm.o io.o $(LIBRABIT) $(LIBDMLC)

# dependency on rabit
subtree/rabit/lib/librabit.a: subtree/rabit/src/engine.cc
	+	cd subtree/rabit;make lib/librabit.a; cd ../..
subtree/rabit/lib/librabit_empty.a: subtree/rabit/src/engine_empty.cc
	+	cd subtree/rabit;make lib/librabit_empty.a; cd ../..
subtree/rabit/lib/librabit_mock.a: subtree/rabit/src/engine_mock.cc
	+	cd subtree/rabit;make lib/librabit_mock.a; cd ../..
subtree/rabit/lib/librabit_mpi.a: subtree/rabit/src/engine_mpi.cc
	+	cd subtree/rabit;make lib/librabit_mpi.a; cd ../..

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS) 

$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS) 

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS) $(DLLFLAGS)

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
	rm -rf xgboost/src/*.o xgboost/src/*.so xgboost/src/*.dll
	rm -rf xgboost/src/*/*.o
	rm -rf subtree/rabit/src/*.o
	rm -rf xgboost/demo/*.model xgboost/demo/*.buffer xgboost/demo/*.txt
	rm -rf xgboost/demo/runall.R
	cp -r src xgboost/src/src
	mkdir xgboost/src/subtree
	mkdir xgboost/src/subtree/rabit
	cp -r subtree/rabit/include xgboost/src/subtree/rabit/include
	cp -r subtree/rabit/src xgboost/src/subtree/rabit/src
	rm -rf xgboost/src/subtree/rabit/src/*.o
	mkdir xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.h xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.cpp xgboost/src/wrapper
	cp ./LICENSE xgboost
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars
	cp xgboost/src/Makevars xgboost/src/Makevars.win
	# R CMD build --no-build-vignettes xgboost
	R CMD build xgboost
	rm -rf xgboost
	R CMD check --as-cran xgboost*.tar.gz

clean:
	$(RM) -rf $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
	cd subtree/rabit; make clean; cd ..
