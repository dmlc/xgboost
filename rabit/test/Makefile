RABIT_BUILD_DMLC = 0

ifeq ($(RABIT_BUILD_DMLC),1)
    DMLC=../dmlc-core
else
    DMLC=../../dmlc-core
endif

MPICXX=../mpich/bin/mpicxx
export LDFLAGS=  -L../lib -pthread -lm
export CFLAGS = -Wall -O3 -Wno-unknown-pragmas

export CC = gcc
export CXX = g++


#----------------------------
# Settings for power and arm arch
#----------------------------
ARCH := $(shell uname -a)
ifneq (,$(filter $(ARCH), armv6l armv7l powerpc64le ppc64le aarch64))
	CFLAGS += -march=native
else
	CFLAGS += -msse2
endif

ifndef WITH_FPIC
	WITH_FPIC = 1
endif
ifeq ($(WITH_FPIC), 1)
	CFLAGS += -fPIC
endif

CFLAGS += -I../include -I $(DMLC)/include -std=c++11

# specify tensor path
BIN = speed_test model_recover local_recover lazy_recover
OBJ = $(RABIT_OBJ) speed_test.o model_recover.o local_recover.o lazy_recover.o
MPIBIN = speed_test.mpi
.PHONY: clean all lib mpi

.PHONY: lib all

all: $(BIN)

lib:
	cd ..;make clean;make;cd -

.PHONY: mpi
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
	$(MPICXX) $(CFLAGS) -I../mpich/include -shared -o $@ $(filter %.cpp %.o %.c %.cc, $^) \
    ../lib/librabit_mpi.so $(LDFLAGS) -L../mpich/lib -Wl,-rpath,../mpich/lib -lmpi

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) *~ ../src/*~
