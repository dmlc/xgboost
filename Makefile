export CC  = gcc
export CXX = g++
export LDFLAGS= -pthread -lm 

ifeq ($(no_omp),1)
	export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas 
else
	export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fopenmp
endif

# specify tensor path
BIN = xgboost
OBJ = io.o
SLIB = python/libxgboostwrapper.so
.PHONY: clean all

all: $(BIN) $(OBJ) $(SLIB)

xgboost: src/xgboost_main.cpp io.o src/data.h src/tree/*.h src/tree/*.hpp src/gbm/*.h src/gbm/*.hpp src/utils/*.h src/learner/*.h src/learner/*.hpp 
io.o: src/io/io.cpp src/data.h src/utils/*.h
# now the wrapper takes in two files. io and wrapper part
python/libxgboostwrapper.so: python/xgboost_wrapper.cpp src/io/io.cpp src/*.h src/*/*.hpp src/*/*.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~ */*~ */*/*~
