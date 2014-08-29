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

.PHONY: clean all python 

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

R-package.tar.gz:
	rm -rf xgboost-R
	cp -r R-package xgboost-R
	rm -rf xgboost-R/src/*.o xgboost-R/src/*.so xgboost-R/src/*.dll
	cp -r src xgboost-R/src/src
	cp -r wrapper xgboost-R/src/wrapper
	cp ./LICENSE xgboost-R
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' > xgboost-R/src/Makevars
	cat R-package/src/Makevars.win|sed '2s/.*/PKGROOT=./' > xgboost-R/src/Makevars.win
	tar czf $@ xgboost-R
	rm -rf xgboost-R

clean:
	$(RM) $(OBJ) $(BIN) $(SLIB) *.o *~ */*~ */*/*~
