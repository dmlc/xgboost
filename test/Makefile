export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# specify tensor path
BIN = test_allreduce test_recover
OBJ = engine_base.o engine_robust.o engine.o
.PHONY: clean all

all: $(BIN) $(MPIBIN)

engine_tcp.o: ../src/engine_tcp.cpp ../src/*.h
engine_base.o: ../src/engine_base.cc ../src/*.h
engine.o: ../src/engine.cc ../src/*.h
engine_robust.o: ../src/engine_robust.cc ../src/*.h
test_allreduce: test_allreduce.cpp ../src/*.h $(OBJ)
test_recover: test_recover.cpp ../src/*.h $(OBJ)

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) *~ ../src/*~
