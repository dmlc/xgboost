export MPICXX = mpicxx
export LDFLAGS=  -L../lib -pthread -lm
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../include  -std=c++11

OS := $(shell uname)

ifeq ($(OS), Darwin)
    ifndef CC
        export CC = $(if $(shell which clang), clang, gcc)
    endif
    ifndef CXX
        export CXX = $(if $(shell which clang++), clang++, g++)
    endif
else
    ifndef CC
        export CC = gcc
    endif
    ifndef CXX
        export CXX = g++
    endif
    LDFLAGS += -lrt
endif

# specify tensor path
BIN = speed_test model_recover local_recover lazy_recover
OBJ = $(RABIT_OBJ) speed_test.o model_recover.o local_recover.o lazy_recover.o
MPIBIN = speed_test.mpi
.PHONY: clean all lib mpi

all: $(BIN)
lib:
	cd ..;make;cd -
mpi:
	cd ..;make mpi;cd -
# programs
speed_test.o: speed_test.cc ../include/rabit/*.h lib mpi
model_recover.o: model_recover.cc ../include/rabit/*.h lib
local_recover.o: local_recover.cc ../include/rabit/*.h lib
lazy_recover.o: lazy_recover.cc ../include/rabit/*.h lib

# we can link against MPI version to get use MPI
speed_test: speed_test.o  $(RABIT_OBJ)
speed_test.mpi: speed_test.o $(MPIOBJ)
model_recover: model_recover.o  $(RABIT_OBJ)
local_recover: local_recover.o  $(RABIT_OBJ)
lazy_recover: lazy_recover.o  $(RABIT_OBJ)

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) ../lib/librabit_mock.a $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) :
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^) $(LDFLAGS) -lrabit_mpi

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) *~ ../src/*~
