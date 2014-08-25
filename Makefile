export CC  = gcc
export CXX = g++
export LDFLAGS= -pthread -lm 
# note for R module
# add include path to Rinternals.h here

ifeq ($(no_omp),1)
	export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -DDISABLE_OPENMP -funroll-loops 
else
	export CFLAGS = -Wall -O3 -msse2 -Wno-unknown-pragmas -fopenmp  -funroll-loops 
endif

# expose these flags to R CMD SHLIB
export PKG_CPPFLAGS = $(CFLAGS)

# specify tensor path
BIN = xgboost
OBJ = 
SLIB = wrapper/libxgboostwrapper.so 
RLIB = wrapper/libxgboostR.so 
.PHONY: clean all R python

all: $(BIN) wrapper/libxgboostwrapper.so
R: wrapper/libxgboostR.so
python: wrapper/libxgboostwrapper.so
xgboost: src/xgboost_main.cpp src/io/io.cpp src/data.h src/tree/*.h src/tree/*.hpp src/gbm/*.h src/gbm/*.hpp src/utils/*.h src/learner/*.h src/learner/*.hpp 
# now the wrapper takes in two files. io and wrapper part
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/io/io.cpp src/*.h src/*/*.hpp src/*/*.h
wrapper/libxgboostR.so: wrapper/xgboost_wrapper.cpp wrapper/xgboost_R.cpp src/io/io.cpp src/*.h src/*/*.hpp src/*/*.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)

$(RLIB) :
	R CMD SHLIB -c -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) $(SLIB) $(RLIB) *~ */*~ */*/*~
