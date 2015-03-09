ifndef CXX
export CXX = g++
endif
export MPICXX = mpicxx
export LDFLAGS= -Llib -lrt
export WARNFLAGS= -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas -pedantic 
export CFLAGS = -O3 -msse2 -fPIC $(WARNFLAGS) 

# build path
BPATH=.
# objectives that makes up rabit library
MPIOBJ= $(BPATH)/engine_mpi.o
OBJ= $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o $(BPATH)/engine_empty.o $(BPATH)/engine_mock.o\
	$(BPATH)/rabit_wrapper.o $(BPATH)/engine_base.o
SLIB= wrapper/librabit_wrapper.so wrapper/librabit_wrapper_mock.so wrapper/librabit_wrapper_mpi.so
ALIB= lib/librabit.a lib/librabit_mpi.a lib/librabit_empty.a lib/librabit_mock.a lib/librabit_base.a
HEADERS=src/*.h include/*.h include/rabit/*.h
.PHONY: clean all install mpi python

all: lib/librabit.a lib/librabit_mock.a  wrapper/librabit_wrapper.so wrapper/librabit_wrapper_mock.so lib/librabit_base.a
mpi: lib/librabit_mpi.a wrapper/librabit_wrapper_mpi.so
python: wrapper/librabit_wrapper.so wrapper/librabit_wrapper_mock.so

$(BPATH)/allreduce_base.o: src/allreduce_base.cc $(HEADERS)
$(BPATH)/engine.o: src/engine.cc $(HEADERS)
$(BPATH)/allreduce_robust.o: src/allreduce_robust.cc $(HEADERS)
$(BPATH)/engine_mpi.o: src/engine_mpi.cc $(HEADERS)
$(BPATH)/engine_empty.o: src/engine_empty.cc $(HEADERS)
$(BPATH)/engine_mock.o: src/engine_mock.cc $(HEADERS)
$(BPATH)/engine_base.o: src/engine_base.cc $(HEADERS)

lib/librabit.a: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o
lib/librabit_base.a: $(BPATH)/allreduce_base.o $(BPATH)/engine_base.o
lib/librabit_mock.a: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine_mock.o
lib/librabit_empty.a: $(BPATH)/engine_empty.o
lib/librabit_mpi.a: $(MPIOBJ)
# wrapper code
$(BPATH)/rabit_wrapper.o: wrapper/rabit_wrapper.cc
wrapper/librabit_wrapper.so: $(BPATH)/rabit_wrapper.o lib/librabit.a
wrapper/librabit_wrapper_mock.so: $(BPATH)/rabit_wrapper.o lib/librabit_mock.a
wrapper/librabit_wrapper_mpi.so: $(BPATH)/rabit_wrapper.o lib/librabit_mpi.a

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) : 
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(ALIB):
	ar cr $@ $+

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(MPIOBJ) $(ALIB) $(MPIALIB) *~ src/*~ include/*~ include/*/*~ wrapper/*~

