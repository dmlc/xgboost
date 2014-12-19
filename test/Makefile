export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm  -lrt
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

# specify tensor path
BIN = speed_test test_model_recover test_local_recover
# objectives that makes up rabit library
RABIT_OBJ = allreduce_base.o allreduce_robust.o engine.o
MPIOBJ = engine_mpi.o

OBJ = $(RABIT_OBJ) speed_test.o test_model_recover.o test_local_recover.o
MPIBIN = speed_test.mpi
.PHONY: clean all

all: $(BIN) $(MPIBIN)
# the rabit library
allreduce_base.o: ../src/allreduce_base.cc ../src/*.h
engine.o: ../src/engine.cc ../src/*.h
engine_mpi.o: ../src/engine_mpi.cc
allreduce_robust.o: ../src/allreduce_robust.cc ../src/*.h

# programs 
speed_test.o: speed_test.cpp ../src/*.h
test_model_recover.o: test_model_recover.cpp ../src/*.h
test_local_recover.o: test_local_recover.cpp ../src/*.h

# we can link against MPI version to get use MPI
speed_test: speed_test.o  $(RABIT_OBJ)
speed_test.mpi: speed_test.o $(MPIOBJ)
test_model_recover: test_model_recover.o  $(RABIT_OBJ)
test_local_recover: test_local_recover.o  $(RABIT_OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS)

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) *~ ../src/*~
