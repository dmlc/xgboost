# this is the common build script for rabit programs
# you do not have to use it 
export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm -L../../lib
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I../../include -I../common

.PHONY: clean all lib mpi
all: $(BIN) $(MOCKBIN)
mpi: $(MPIBIN)

lib:
	cd ../..;make lib/librabit.a lib/librabit_mock.a; cd -
libmpi:
	cd ../..;make lib/librabit_mpi.a;cd -

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
