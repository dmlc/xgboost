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

ifeq ($(OS), Windows_NT)
	UNAME="Windows"
else
	UNAME=$(shell uname)
endif

include $(config)
ifeq ($(USE_OPENMP), 0)
	export NO_OPENMP = 1
endif
include $(DMLC_CORE)/make/dmlc.mk

# include the plugins
include $(XGB_PLUGINS)

# use customized config file
ifndef CC
export CC  = $(if $(shell which gcc-6),gcc-6,gcc)
endif
ifndef CXX
export CXX = $(if $(shell which g++-6),g++-6,g++)
endif

# on Mac OS X, force brew gcc-6, since the Xcode c++ fails anyway
# it is useful for pip install compiling-on-the-fly
OS := $(shell uname)
ifeq ($(OS), Darwin)
export CC = $(if $(shell which gcc-6),gcc-6,$(if $(shell which gcc-mp-6), gcc-mp-6, clang))
export CXX = $(if $(shell which g++-6),g++-6,$(if $(shell which g++-mp-6),g++-mp-6, clang++))

endif

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS) $(DMLC_LDFLAGS) $(PLUGIN_LDFLAGS)
export CFLAGS=  -std=c++0x -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops -Iinclude $(ADD_CFLAGS) $(PLUGIN_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include
#java include path
export JAVAINCFLAGS = -I${JAVA_HOME}/include -I./java

ifeq ($(OS), FreeBSD)
export CC  = $(if $(shell which gcc6),gcc6)
export CXX = $(if $(shell which g++6),g++6)

LDFLAGS += -Wl,-rpath=/usr/local/lib/gcc6
export MAKE = gmake

endif

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifneq ($(UNAME), Windows)
	CFLAGS += -fPIC
	XGBOOST_DYLIB = lib/libxgboost.so
else
	XGBOOST_DYLIB = lib/libxgboost.dll
	JAVAINCFLAGS += -I${JAVA_HOME}/include/win32
endif

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
	JAVAINCFLAGS += -I${JAVA_HOME}/include/linux
endif

ifeq ($(UNAME), Darwin)
	JAVAINCFLAGS += -I${JAVA_HOME}/include/darwin
endif

ifeq ($(UNAME), FreeBSD)
	JAVAINCFLAGS += -I${JAVA_HOME}/include/freebsd
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
else
	CFLAGS += -DDISABLE_OPENMP
endif


# specify tensor path
.PHONY: clean all lint clean_all doxygen rcpplint pypack Rpack Rbuild Rcheck java pylint


all: lib/libxgboost.a $(XGBOOST_DYLIB) xgboost

$(DMLC_CORE)/libdmlc.a: $(wildcard $(DMLC_CORE)/src/*.cc $(DMLC_CORE)/src/*/*.cc)
	+ cd $(DMLC_CORE); $(MAKE) libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

$(RABIT)/lib/$(LIB_RABIT): $(wildcard $(RABIT)/src/*.cc)
	+ cd $(RABIT); $(MAKE) lib/$(LIB_RABIT); cd $(ROOTDIR)

jvm: jvm-packages/lib/libxgboost4j.so

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC)) $(PLUGIN_OBJS)
AMALGA_OBJ = amalgamation/xgboost-all0.o
LIB_DEP = $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/$(LIB_RABIT)
ALL_DEP = $(filter-out build/cli_main.o, $(ALL_OBJ)) $(LIB_DEP)
CLI_OBJ = build/cli_main.o

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

build_plugin/%.o: plugin/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build_plugin/$*.o $< >build_plugin/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

# The should be equivalent to $(ALL_OBJ)  except for build/cli_main.o
amalgamation/xgboost-all0.o: amalgamation/xgboost-all0.cc
	$(CXX) -c $(CFLAGS) $< -o $@

# Equivalent to lib/libxgboost_all.so
lib/libxgboost_all.so: $(AMALGA_OBJ) $(LIB_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libxgboost.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libxgboost.dll lib/libxgboost.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %a,  $^) $(LDFLAGS)

jvm-packages/lib/libxgboost4j.so: jvm-packages/xgboost4j/src/native/xgboost4j.cpp $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(JAVAINCFLAGS) -shared -o $@ $(filter %.cpp %.o %.a, $^) $(LDFLAGS)

xgboost: $(CLI_OBJ) $(ALL_DEP)
	$(CXX) $(CFLAGS) -o $@  $(filter %.o %.a, $^)  $(LDFLAGS)

rcpplint:
	python2 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} R-package/src

lint: rcpplint
	python2 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} include src plugin python-package

pylint:
	flake8 --ignore E501 python-package
	flake8 --ignore E501 tests/python
clean:
	$(RM) -rf build build_plugin lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o xgboost

clean_all: clean
	cd $(DMLC_CORE); $(MAKE) clean; cd $(ROOTDIR)
	cd $(RABIT); $(MAKE) clean; cd $(ROOTDIR)

doxygen:
	doxygen doc/Doxyfile

# create standalone python tar file.
pypack: ${XGBOOST_DYLIB}
	cp ${XGBOOST_DYLIB} python-package/xgboost
	cd python-package; tar cf xgboost.tar xgboost; cd ..

# create pip installation pack for PyPI
pippack:
	$(MAKE) clean_all
	rm -rf xgboost-python
	cp -r python-package xgboost-python
	cp -r Makefile xgboost-python/xgboost/
	cp -r make xgboost-python/xgboost/
	cp -r src xgboost-python/xgboost/
	cp -r include xgboost-python/xgboost/
	cp -r dmlc-core xgboost-python/xgboost/
	cp -r rabit xgboost-python/xgboost/

# Script to make a clean installable R package.
Rpack:
	$(MAKE) clean_all
	rm -rf xgboost xgboost*.tar.gz
	cp -r R-package xgboost
	rm -rf xgboost/src/*.o xgboost/src/*.so xgboost/src/*.dll
	rm -rf xgboost/src/*/*.o
	rm -rf xgboost/demo/*.model xgboost/demo/*.buffer xgboost/demo/*.txt
	rm -rf xgboost/demo/runall.R
	cp -r src xgboost/src/src
	cp -r include xgboost/src/include
	cp -r amalgamation xgboost/src/amalgamation
	mkdir -p xgboost/src/rabit
	cp -r rabit/include xgboost/src/rabit/include
	cp -r rabit/src xgboost/src/rabit/src
	rm -rf xgboost/src/rabit/src/*.o
	mkdir -p xgboost/src/dmlc-core
	cp -r dmlc-core/include xgboost/src/dmlc-core/include
	cp -r dmlc-core/src xgboost/src/dmlc-core/src
	cp ./LICENSE xgboost
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' | sed '3s/.*/ENABLE_STD_THREAD=0/' > xgboost/src/Makevars
	cp xgboost/src/Makevars xgboost/src/Makevars.win

Rbuild:
	$(MAKE) Rpack
	R CMD build --no-build-vignettes xgboost
	rm -rf xgboost

Rcheck:
	$(MAKE) Rbuild
	R CMD check  xgboost*.tar.gz

-include build/*.d
-include build/*/*.d
-include build_plugin/*/*.d
