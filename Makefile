ifndef config
ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = dmlc-core
endif

ifndef RABIT
	RABIT = rabit
endif

ROOTDIR = $(CURDIR)
UNAME= $(shell uname)

include $(config)
ifeq ($(USE_OPENMP), 0)
	export NO_OPENMP = 1
endif
include $(DMLC_CORE)/make/dmlc.mk

# use customized config file
ifndef CC
export CC  = $(if $(shell which gcc-5),gcc-5,gcc)
endif
ifndef CXX
export CXX = $(if $(shell which g++-5),g++-5,g++)
endif

ifeq ($(OS), Windows_NT)
	export CXX = g++ -m64
	export CC = gcc -m64
endif

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS) $(DMLC_LDFLAGS)
export CFLAGS= -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops -fPIC -Iinclude $(ADD_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
else
	CFLAGS += -DDISABLE_OPENMP
endif

# specify tensor path
.PHONY: clean all lint clean_all

all: lib/libxgboost.a lib/libxgboost.so

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

$(RABIT)/lib/$(LIB_RABIT):
	+ cd $(RABIT); make lib/$(LIB_RABIT); cd $(ROOTDIR)

SRC = $(wildcard src/*.cc src/*/*.cc)
OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
LIB_DEP = $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/$(LIB_RABIT)
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
	cd $(RABIT); make clean; cd -

-include build/*.d
-include build/*/*.d
