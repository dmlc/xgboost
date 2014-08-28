export CC  = gcc
export CXX = g++
export LDFLAGS= -pthread -lm 

export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# specify tensor path
BIN = xgboost
OBJ = updater.o gbm.o io.o
SLIB = wrapper/libxgboostwrapper.so 
#RLIB = wrapper/libxgboostR.so 
.PHONY: clean all R python

all: $(BIN) $(OBJ)
#python: wrapper/libxgboostwrapper.so
#xgboost: src/xgboost_main.cpp src/io/io.cpp src/data.h src/tree/*.h src/tree/*.hpp src/gbm/*.h src/gbm/*.hpp src/utils/*.h src/learner/*.h src/learner/*.hpp 

# now the wrapper takes in two files. io and wrapper part
wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp $(OBJ)
updater.o: src/tree/updater.cpp 
gbm.o: src/gbm/gbm.cpp 
io.o: src/io/io.cpp
xgboost: src/xgboost_main.cpp $(OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) $(SLIB) $(RLIB) *~ */*~ */*/*~
