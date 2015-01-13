export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm  -lrt -L../lib
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include  -std=c++11

# specify tensor path
BIN = speed_test test_model_recover test_local_recover
OBJ = $(RABIT_OBJ) speed_test.o test_model_recover.o test_local_recover.o
MPIBIN = speed_test.mpi
.PHONY: clean all lib mpi

all: $(BIN) $(MPIBIN)
lib:
	cd ..;make;cd -
mpi:
	cd ..;make mpi;cd -
# programs 
speed_test.o: speed_test.cpp ../include/*.h lib mpi
test_model_recover.o: test_model_recover.cpp ../include/*.h lib
test_local_recover.o: test_local_recover.cpp ../include/*.h lib

# we can link against MPI version to get use MPI
speed_test: speed_test.o  $(RABIT_OBJ)
speed_test.mpi: speed_test.o $(MPIOBJ)
test_model_recover: test_model_recover.o  $(RABIT_OBJ)
test_local_recover: test_local_recover.o  $(RABIT_OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS) -lrabit_mock

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS) -lrabit_mpi

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) *~ ../src/*~
