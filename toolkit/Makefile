export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include 

# specify tensor path
BIN = kmeans.rabit
# objectives that makes up rabit library
OBJ = kmeans.o
MPIBIN = kmeans.mpi
.PHONY: clean all lib libmpi

all: $(BIN)

lib:
	cd ..;make lib/librabit.a;cd -
libmpi:
	cd ..;make lib/librabit_mpi.a;cd -

kmeans.o: kmeans.cpp ../src/*.h

# we can link against MPI version to get use MPI
kmeans.rabit: kmeans.o lib
kmeans.mpi: kmeans.o libmpi

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^)  $(LDFLAGS) -lrabit_mpi 

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~
