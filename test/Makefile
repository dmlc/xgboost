export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm  -lrt
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

# specify tensor path
BIN = test_allreduce test_recover test_model_recover speed_test
# objectives that makes up rabit library
RABIT_OBJ = allreduce_base.o allreduce_robust.o engine.o
MPIOBJ = engine_mpi.o

OBJ = $(RABIT_OBJ) test_allreduce.o test_recover.o test_model_recover.o speed_test.o
MPIBIN = test_allreduce.mpi speed_test.mpi
.PHONY: clean all

all: $(BIN) $(MPIBIN)

allreduce_base.o: ../src/allreduce_base.cc ../src/*.h
engine.o: ../src/engine.cc ../src/*.h
allreduce_robust.o: ../src/allreduce_robust.cc ../src/*.h
engine_mpi.o: ../src/engine_mpi.cc
test_allreduce.o: test_allreduce.cpp ../src/*.h
speed_test.o: speed_test.cpp ../src/*.h
test_recover.o: test_recover.cpp ../src/*.h 
test_model_recover.o: test_model_recover.cpp ../src/*.h

# we can link against MPI version to get use MPI
test_allreduce: test_allreduce.o  $(RABIT_OBJ)
test_allreduce.mpi: test_allreduce.o $(MPIOBJ)
speed_test: speed_test.o  $(RABIT_OBJ)
speed_test.mpi: speed_test.o $(MPIOBJ)
test_recover: test_recover.o  $(RABIT_OBJ)
test_model_recover: test_model_recover.o  $(RABIT_OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS)

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~
