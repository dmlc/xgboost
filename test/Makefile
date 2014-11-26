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
BIN = test_allreduce
OBJ = engine_tcp.o
.PHONY: clean all

all: $(BIN) $(MPIBIN)

engine_tcp.o: ../src/engine_tcp.cpp ../src/*.h
test_allreduce: test_allreduce.cpp ../src/*.h engine_tcp.o

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

clean:
	$(RM) $(BIN) $(MPIBIN) *~ ../src/*~
