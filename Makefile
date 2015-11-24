ifndef CC
export CC  = $(if $(shell which gcc-5),gcc-5,gcc)
endif
ifndef CXX
export CXX = $(if $(shell which g++-5),g++-5,g++)
endif

export LDFLAGS= -pthread -lm
export CFLAGS= -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops -fPIC -Iinclude

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifeq ($(OS), Windows_NT)
	export CXX = g++ -m64
	export CC = gcc -m64
endif

UNAME= $(shell uname)

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
endif

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP
else
	ifeq ($(omp_mac_static),1)
		CFLAGS += -static-libgcc -static-libstdc++ -L. -fopenmp
	else
		CFLAGS += -fopenmp
	endif
endif


ifndef DMLC_CORE
	DMLC_CORE = dmlc-core
endif

ifndef RABIT
	RABIT = rabit
endif

# specify tensor path
.PHONY: clean all lint

all: lib/libxgboost.a lib/libxgboost.so

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a; cd $(ROOTDIR)

$(RABIT)/lib/librabit.a:
	+ cd $(RABIT); make lib/librabit.a; cd $(ROOTDIR)

CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include
Lib_DEP = $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/librabit.a

SRC = $(wildcard src/*.cc src/*/*.cc)
OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
LIB_DEP += $(DMLC_CORE)/libdmlc.a
ALL_DEP = $(OBJ) $(LIB_DEP)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -std=c++0x -c $(CFLAGS) -c $< -o $@

lib/libxgboost.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libxgboost.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lint:
	python2 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} include src

clean:
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~

clean_all: clean
	cd $(DMLC_CORE); make clean; cd -
	cd $(PS_PATH); make clean; cd -

-include build/*.d
-include build/*/*.d
