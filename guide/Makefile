export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include 

.PHONY: clean all lib libmpi
BIN = basic.rabit
MOCKBIN= 

all: $(BIN) $(MOCKBIN)
basic.rabit: basic.cc lib

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit
$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit_mock

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^)  $(LDFLAGS) -lrabit_mpi 

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~