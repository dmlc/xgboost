export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS=
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fPIC -I../src

BPATH=lib
# objectives that makes up rabit library
MPIOBJ= $(BPATH)/engine_mpi.o
OBJ= $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o $(BPATH)/engine_empty.o
ALIB= lib/librabit.a lib/librabit_mpi.a lib/librabit_empty.a

.PHONY: clean all

all: $(ALIB)

$(BPATH)/allreduce_base.o: src/allreduce_base.cc src/*.h
$(BPATH)/engine.o: src/engine.cc src/*.h
$(BPATH)/allreduce_robust.o: src/allreduce_robust.cc src/*.h
$(BPATH)/engine_mpi.o: src/engine_mpi.cc src/*.h
$(BPATH)/engine_empty.o: src/engine_empty.cc src/*.h

lib/librabit.a: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o
lib/librabit_empty.a: $(BPATH)/engine_empty.o
lib/librabit_mpi.a: $(MPIOBJ)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(ALIB):
	ar cr $@ $+

clean:
	$(RM) $(OBJ) $(MPIOBJ) $(ALIB) $(MPIALIB) *~ src/*~
