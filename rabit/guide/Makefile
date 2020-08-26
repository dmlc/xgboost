export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2 -std=c++11 -Wno-unknown-pragmas -fPIC -fopenmp -I../include

.PHONY: clean all lib libmpi
BIN = basic.rabit broadcast.rabit
MOCKBIN= lazy_allreduce.mock

all: $(BIN)
basic.rabit: basic.cc lib ../lib/librabit.a
broadcast.rabit: broadcast.cc lib ../lib/librabit.a
lazy_allreduce.mock: lazy_allreduce.cc lib ../lib/librabit.a

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

$(MOCKBIN) :
	$(CXX) $(CFLAGS) -std=c++11 -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit_mock

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean:
	$(RM) $(OBJ) $(BIN) $(MOCKBIN) *~ ../src/*~
