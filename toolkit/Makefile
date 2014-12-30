export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include 

# specify tensor path
BIN = kmeans.rabit 
MOCKBIN= kmeans.mock
MPIBIN = kmeans.mpi
# objectives that makes up rabit library
OBJ = kmeans.o
.PHONY: clean all lib 
all: $(BIN)

lib:
	cd ..;make lib/librabit.a lib/librabit_mock.a; cd -
libmpi:
	cd ..;make lib/librabit_mpi.a;cd -

kmeans.o: kmeans.cpp ../src/*.h

# we can link against MPI version to get use MPI
kmeans.rabit: kmeans.o lib
kmeans.mock: kmeans.o lib
kmeans.mpi: kmeans.o libmpi

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit
$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) $(LDFLAGS) -lrabit_mock

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^)  $(LDFLAGS) -lrabit_mpi 

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MOCKBIN) *~ ../src/*~
