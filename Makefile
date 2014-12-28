export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS=
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -Iinclude

BPATH=lib
# objectives that makes up rabit library
MPIOBJ= $(BPATH)/engine_mpi.o
OBJ= $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o $(BPATH)/engine_empty.o $(BPATH)/engine_mock.o
ALIB= lib/librabit.a lib/librabit_mpi.a lib/librabit_empty.a lib/librabit_mock.a
HEADERS=src/*.h include/*.h include/rabit/*.h
.PHONY: clean all

all: $(ALIB)

$(BPATH)/allreduce_base.o: src/allreduce_base.cc $(HEADERS)
$(BPATH)/engine.o: src/engine.cc $(HEADERS)
$(BPATH)/allreduce_robust.o: src/allreduce_robust.cc $(HEADERS)
$(BPATH)/engine_mpi.o: src/engine_mpi.cc $(HEADERS)
$(BPATH)/engine_empty.o: src/engine_empty.cc $(HEADERS)
$(BPATH)/engine_mock.o: src/engine_mock.cc $(HEADERS)

lib/librabit.a: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o
lib/librabit_mock.a: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine_mock.o
lib/librabit_empty.a: $(BPATH)/engine_empty.o
lib/librabit_mpi.a: $(MPIOBJ)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(ALIB):
	ar cr $@ $+

clean:
	$(RM) $(OBJ) $(MPIOBJ) $(ALIB) $(MPIALIB) *~ src/*~ include/*~ include/*/*~
