export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

# specify tensor path
BIN = kmeans
# objectives that makes up rabit library
RABIT_OBJ = allreduce_base.o allreduce_robust.o engine.o
MPIOBJ = engine_mpi.o

OBJ = $(RABIT_OBJ) kmeans.o
MPIBIN = kmeans.mpi
.PHONY: clean all

all: $(BIN) $(MPIBIN)

allreduce_base.o: ../src/allreduce_base.cc ../src/*.h
engine.o: ../src/engine.cc ../src/*.h
allreduce_robust.o: ../src/allreduce_robust.cc ../src/*.h
engine_mpi.o: ../src/engine_mpi.cc
kmeans.o: kmeans.cpp ../src/*.h

# we can link against MPI version to get use MPI
kmeans: kmeans.o  $(RABIT_OBJ)
kmeans.mpi: kmeans.o $(MPIOBJ)
 
$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS)  $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^)

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~
