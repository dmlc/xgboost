export CC  = gcc
export CXX = g++
export LDFLAGS= -pthread -lm 

export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -pedantic 

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# specify tensor path
BIN = xgboost
OBJ = updater.o gbm.o io.o
SLIB = wrapper/libxgboostwrapper.so 

.PHONY: clean all python Rpack

all: $(BIN) $(OBJ) $(SLIB) 

python: wrapper/libxgboostwrapper.so
# now the wrapper takes in two files. io and wrapper part
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp $(OBJ)
updater.o: src/tree/updater.cpp  src/tree/*.hpp src/*.h src/tree/*.h
gbm.o: src/gbm/gbm.cpp src/gbm/*.hpp src/gbm/*.h
io.o: src/io/io.cpp src/io/*.hpp src/utils/*.h src/learner/dmatrix.h src/*.h
xgboost: src/xgboost_main.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h $(OBJ)
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h $(OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

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
	$(RM) $(OBJ) $(BIN) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
