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
BIN = test_group_data

.PHONY: clean all

all: $(BIN) $(MPIBIN)

test_group_data: test_group_data.cpp

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

clean:
	$(RM) $(BIN) $(MPIBIN) *~
