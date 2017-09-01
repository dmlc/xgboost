OS := $(shell uname)

ifeq ($(OS), Linux)
ifndef CXX
export CXX = g++
endif
export MPICXX = mpicxx
export LDFLAGS= -Llib -lrt

endif

ifeq ($(OS), FreeBSD)
ifndef CXX
export CXX = g++6
endif
export MPICXX = /usr/local/mpi/bin/mpicxx
export LDFLAGS= -Llib -Wl,-rpath=/usr/local/lib/gcc6

endif

export WARNFLAGS= -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas -std=c++0x
export CFLAGS = -O3 $(WARNFLAGS)

#----------------------------
# Settings for power and arm arch
#----------------------------
ARCH := $(shell uname -a)
ifneq (,$(filter $(ARCH), powerpc64le ppc64le ))
	USE_SSE=0
else
	USE_SSE=1
endif

ifndef USE_SSE
	USE_SSE = 1
endif

ifeq ($(USE_SSE), 1)
	CFLAGS += -msse2
endif

ifndef WITH_FPIC
	WITH_FPIC = 1
endif
ifeq ($(WITH_FPIC), 1)
	CFLAGS += -fPIC
endif

ifndef LINT_LANG
	LINT_LANG="all"
endif

# build path
BPATH=.
# objectives that makes up rabit library
MPIOBJ= $(BPATH)/engine_mpi.o
OBJ= $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o $(BPATH)/engine_empty.o $(BPATH)/engine_mock.o\
	$(BPATH)/c_api.o $(BPATH)/engine_base.o
SLIB= lib/librabit.so lib/librabit_mpi.so lib/librabit_mock.so lib/librabit_base.so
ALIB= lib/librabit.a lib/librabit_mpi.a lib/librabit_empty.a lib/librabit_mock.a lib/librabit_base.a
HEADERS=src/*.h include/rabit/*.h include/rabit/internal/*.h
DMLC=dmlc-core

.PHONY: clean all install mpi python lint doc doxygen

all: lib/librabit.a lib/librabit_mock.a  lib/librabit.so lib/librabit_base.a lib/librabit_mock.so
mpi: lib/librabit_mpi.a lib/librabit_mpi.so

$(BPATH)/allreduce_base.o: src/allreduce_base.cc $(HEADERS)
$(BPATH)/engine.o: src/engine.cc $(HEADERS)
$(BPATH)/allreduce_robust.o: src/allreduce_robust.cc $(HEADERS)
$(BPATH)/engine_mpi.o: src/engine_mpi.cc $(HEADERS)
$(BPATH)/engine_empty.o: src/engine_empty.cc $(HEADERS)
$(BPATH)/engine_mock.o: src/engine_mock.cc $(HEADERS)
$(BPATH)/engine_base.o: src/engine_base.cc $(HEADERS)
$(BPATH)/c_api.o: src/c_api.cc $(HEADERS)

lib/librabit.a lib/librabit.so: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine.o $(BPATH)/c_api.o
lib/librabit_base.a lib/librabit_base.so: $(BPATH)/allreduce_base.o $(BPATH)/engine_base.o $(BPATH)/c_api.o
lib/librabit_mock.a lib/librabit_mock.so: $(BPATH)/allreduce_base.o $(BPATH)/allreduce_robust.o $(BPATH)/engine_mock.o $(BPATH)/c_api.o
lib/librabit_empty.a: $(BPATH)/engine_empty.o $(BPATH)/c_api.o
lib/librabit_mpi.a lib/librabit_mpi.so: $(MPIOBJ)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) :
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(ALIB):
	ar cr $@ $+

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS)

lint:
	$(DMLC)/scripts/lint.py rabit $(LINT_LANG) src include

doc doxygen:
	cd include; doxygen ../doc/Doxyfile; cd -

clean:
	$(RM) $(OBJ) $(MPIOBJ) $(ALIB) $(MPIALIB) $(SLIB) *~ src/*~ include/*~ include/*/*~
