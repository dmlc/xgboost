export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

# specify tensor path
BIN = kmeans
# objectives that makes up rabit library
RABIT_OBJ = allreduce_base.o allreduce_robust.o engine.o
MPIOBJ = engine_mpi.o

OBJ = $(RABIT_OBJ) kmeans.o
MPIBIN = kmeans.mpi
.PHONY: clean all lib

all: $(BIN) $(MPIBIN) 

lib:
	cd ..;make;cd -

kmeans.o: kmeans.cpp ../src/*.h

# we can link against MPI version to get use MPI
kmeans: kmeans.o lib
kmeans.mpi: kmeans.o lib

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^)  $(LDFLAGS) -lrabit_mpi 

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~
