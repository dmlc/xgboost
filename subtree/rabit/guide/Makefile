export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include 

.PHONY: clean all lib libmpi
BIN = basic.rabit broadcast.rabit
MOCKBIN= lazy_allreduce.mock

all: $(BIN)
basic.rabit: basic.cc lib
broadcast.rabit: broadcast.cc lib
lazy_allreduce.mock: lazy_allreduce.cc lib

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit

$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -std=c++11 -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit_mock

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean:
	$(RM) $(OBJ) $(BIN) $(MOCKBIN) *~ ../src/*~