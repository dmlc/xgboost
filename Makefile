export CC  = gcc
export CXX = g++
export LDFLAGS= -pthread -lm 
# note for R module
# add include path to Rinternals.h here
export CPLUS_INCLUDE_PATH=/usr/share/R/include

ifeq ($(no_omp),1)
	export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -DDISABLE_OPENMP 
else
	export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fopenmp
endif

# specify tensor path
BIN = xgboost
OBJ = 
SLIB = wrapper/libxgboostwrapper.so wrapper/libxgboostR.so
.PHONY: clean all R

all: $(BIN) wrapper/libxgboostwrapper.so
R: wrapper/libxgboostR.so

xgboost: src/xgboost_main.cpp src/io/io.cpp src/data.h src/tree/*.h src/tree/*.hpp src/gbm/*.h src/gbm/*.hpp src/utils/*.h src/learner/*.h src/learner/*.hpp 
# now the wrapper takes in two files. io and wrapper part
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/io/io.cpp src/*.h src/*/*.hpp src/*/*.h
wrapper/libxgboostR.so: wrapper/xgboost_wrapper.cpp wrapper/xgboost_R.cpp src/io/io.cpp src/*.h src/*/*.hpp src/*/*.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) $(SLIB) *~ */*~ */*/*~
