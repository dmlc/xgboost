export CC  = gcc
export CXX = clang++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
endif

# specify tensor path
BIN = test_group_data test_quantile test_allreduce
OBJ = sync_tcp.o
.PHONY: clean all

all: $(BIN) $(MPIBIN)

sync_tcp.o: ../src/sync/sync_tcp.cpp ../src/utils/*.h

test_group_data: test_group_data.cpp ../src/utils/*.h
test_quantile: test_quantile.cpp ../src/utils/*.h
test_allreduce: test_allreduce.cpp ../src/utils/*.h ../src/sync/sync.h sync_tcp.o
$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

clean:
	$(RM) $(BIN) $(MPIBIN) *~
