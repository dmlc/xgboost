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

# workarounds for some buggy old make & msys2 versions seen in windows
ifeq (NA, $(shell test ! -d "$(ROOTDIR)" && echo NA ))
        $(warning Attempting to fix non-existing ROOTDIR [$(ROOTDIR)])
        ROOTDIR := $(shell pwd)
        $(warning New ROOTDIR [$(ROOTDIR)] $(shell test -d "$(ROOTDIR)" && echo " is OK" ))
endif
MAKE_OK := $(shell "$(MAKE)" -v 2> /dev/null)
ifndef MAKE_OK
        $(warning Attempting to recover non-functional MAKE [$(MAKE)])
        MAKE := $(shell which make 2> /dev/null)
        MAKE_OK := $(shell "$(MAKE)" -v 2> /dev/null)
endif
$(warning MAKE [$(MAKE)] - $(if $(MAKE_OK),checked OK,PROBLEM))

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
ifdef XGB_PLUGINS
include $(XGB_PLUGINS)
endif

# set compiler defaults for OSX versus *nix
# let people override either
OS := $(shell uname)
ifeq ($(OS), Darwin)
ifndef CC
export CC = $(if $(shell which clang), clang, gcc)
endif
ifndef CXX
export CXX = $(if $(shell which clang++), clang++, g++)
endif
else
# linux defaults
ifndef CC
export CC = gcc
endif
ifndef CXX
export CXX = g++
endif
endif

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS) $(DMLC_LDFLAGS) $(PLUGIN_LDFLAGS)
export CFLAGS= -DDMLC_LOG_CUSTOMIZE=1 -std=c++11 -Wall -Wno-unknown-pragmas -Iinclude $(ADD_CFLAGS) $(PLUGIN_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include -I$(GTEST_PATH)/include
#java include path
export JAVAINCFLAGS = -I${JAVA_HOME}/include -I./java

ifeq ($(TEST_COVER), 1)
	CFLAGS += -g -O0 -fprofile-arcs -ftest-coverage
else
	CFLAGS += -O3 -funroll-loops
ifeq ($(USE_SSE), 1)
	CFLAGS += -msse2
endif
endif

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifeq ($(UNAME), Windows)
	XGBOOST_DYLIB = lib/xgboost.dll
	JAVAINCFLAGS += -I${JAVA_HOME}/include/win32
else
ifeq ($(UNAME), Darwin)
	XGBOOST_DYLIB = lib/libxgboost.dylib
	CFLAGS += -fPIC
else
	XGBOOST_DYLIB = lib/libxgboost.so
	CFLAGS += -fPIC
endif
endif

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
	JAVAINCFLAGS += -I${JAVA_HOME}/include/linux
endif

ifeq ($(UNAME), Darwin)
	JAVAINCFLAGS += -I${JAVA_HOME}/include/darwin
endif

OPENMP_FLAGS =
ifeq ($(USE_OPENMP), 1)
	OPENMP_FLAGS = -fopenmp
else
	OPENMP_FLAGS = -DDISABLE_OPENMP
endif
CFLAGS += $(OPENMP_FLAGS)

# specify tensor path
.PHONY: clean all lint clean_all doxygen rcpplint pypack Rpack Rbuild Rcheck java pylint

all: lib/libxgboost.a $(XGBOOST_DYLIB) xgboost

$(DMLC_CORE)/libdmlc.a: $(wildcard $(DMLC_CORE)/src/*.cc $(DMLC_CORE)/src/*/*.cc)
	+ cd $(DMLC_CORE); "$(MAKE)" libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

$(RABIT)/lib/$(LIB_RABIT): $(wildcard $(RABIT)/src/*.cc)
	+ cd $(RABIT); "$(MAKE)" lib/$(LIB_RABIT) USE_SSE=$(USE_SSE); cd $(ROOTDIR)

jvm: jvm-packages/lib/libxgboost4j.so

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC)) $(PLUGIN_OBJS)
AMALGA_OBJ = amalgamation/xgboost-all0.o
LIB_DEP = $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/$(LIB_RABIT)
ALL_DEP = $(filter-out build/cli_main.o, $(ALL_OBJ)) $(LIB_DEP)
CLI_OBJ = build/cli_main.o
include tests/cpp/xgboost_test.mk

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

lib/xgboost.dll lib/libxgboost.so lib/libxgboost.dylib: $(ALL_DEP)
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

test: $(ALL_TEST)
	$(ALL_TEST)

check: test
	./tests/cpp/xgboost_test

ifeq ($(TEST_COVER), 1)
cover: check
	@- $(foreach COV_OBJ, $(COVER_OBJ), \
		gcov -pbcul -o $(shell dirname $(COV_OBJ)) $(COV_OBJ) > gcov.log || cat gcov.log; \
	)
endif

clean:
	$(RM) -rf build build_plugin lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o #xgboost
	$(RM) -rf build_tests *.gcov tests/cpp/xgboost_test
	if [ -d "R-package/src" ]; then \
		cd R-package/src; \
		$(RM) -rf rabit src include dmlc-core amalgamation *.so *.dll; \
		cd $(ROOTDIR); \
	fi

clean_all: clean
	cd $(DMLC_CORE); "$(MAKE)" clean; cd $(ROOTDIR)
	cd $(RABIT); "$(MAKE)" clean; cd $(ROOTDIR)

doxygen:
	doxygen doc/Doxyfile

# create standalone python tar file.
pypack: ${XGBOOST_DYLIB}
	cp ${XGBOOST_DYLIB} python-package/xgboost
	cd python-package; tar cf xgboost.tar xgboost; cd ..

# create pip source dist (sdist) pack for PyPI
pippack: clean_all
	rm -rf xgboost-python
# remove symlinked directories in python-package/xgboost
	rm -rf python-package/xgboost/lib
	rm -rf python-package/xgboost/dmlc-core
	rm -rf python-package/xgboost/include
	rm -rf python-package/xgboost/make
	rm -rf python-package/xgboost/rabit
	rm -rf python-package/xgboost/src
	cp -r python-package xgboost-python
	cp -r Makefile xgboost-python/xgboost/
	cp -r make xgboost-python/xgboost/
	cp -r src xgboost-python/xgboost/
	cp -r tests xgboost-python/xgboost/
	cp -r include xgboost-python/xgboost/
	cp -r dmlc-core xgboost-python/xgboost/
	cp -r rabit xgboost-python/xgboost/
# Use setup_pip.py instead of setup.py
	mv xgboost-python/setup_pip.py xgboost-python/setup.py
# Build sdist tarball
	cd xgboost-python; python setup.py sdist; mv dist/*.tar.gz ..; cd ..

# Script to make a clean installable R package.
Rpack: clean_all
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
	cat R-package/src/Makevars.in|sed '2s/.*/PKGROOT=./' | sed '3s/.*/ENABLE_STD_THREAD=0/' > xgboost/src/Makevars.in
	cp xgboost/src/Makevars.in xgboost/src/Makevars.win
	sed -i -e 's/@OPENMP_CXXFLAGS@/$$\(SHLIB_OPENMP_CFLAGS\)/g' xgboost/src/Makevars.win
	bash R-package/remove_warning_suppression_pragma.sh
	rm xgboost/remove_warning_suppression_pragma.sh

Rbuild: Rpack
	R CMD build --no-build-vignettes xgboost
	rm -rf xgboost

Rcheck: Rbuild
	R CMD check xgboost*.tar.gz

-include build/*.d
-include build/*/*.d
-include build_plugin/*/*.d
