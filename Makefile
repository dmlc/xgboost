# flags by plugin
PLUGIN_OBJS=
PLUGIN_LDFLAGS=
PLUGIN_CFLAGS=

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

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS) $(DMLC_LDFLAGS) $(PLUGIN_LDFLAGS)
export CFLAGS=  -std=c++0x -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops -fPIC -Iinclude $(ADD_CFLAGS) $(PLUGIN_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include
#java include path
export JAVAINCFLAGS = -I${JAVA_HOME}/include -I./java

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
	JAVAINCFLAGS += -I${JAVA_HOME}/include/linux
endif

ifeq ($(UNAME), Darwin)
	JAVAINCFLAGS += -I${JAVA_HOME}/include/darwin
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
else
	CFLAGS += -DDISABLE_OPENMP
endif


# specify tensor path
.PHONY: clean all lint clean_all rcpplint Rpack Rbuild Rcheck

all: lib/libxgboost.a lib/libxgboost.so xgboost

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

$(RABIT)/lib/$(LIB_RABIT):
	+ cd $(RABIT); make lib/$(LIB_RABIT); cd $(ROOTDIR)

java: java/libxgboost4j.so

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC)) $(PLUGIN_OBJS)
AMALGA_OBJ = amalgamation/xgboost-all0.o
LIB_DEP = $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/$(LIB_RABIT)
ALL_DEP = $(filter-out build/cli_main.o, $(ALL_OBJ)) $(LIB_DEP)
CLI_OBJ = build/cli_main.o

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

build_plugin/%.o: plugin/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build_plugin/$*.o $< >build_plugin/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

# The should be equivalent to $(ALL_OBJ)  except for build/cli_main.o
amalgamation/xgboost-all0.o: amalgamation/xgboost-all0.cc
	$(CXX) -c $(CFLAGS) -c $< -o $@

# Equivalent to lib/libxgboost_all.so
lib/libxgboost_all.so: $(AMALGA_OBJ) $(LIB_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libxgboost.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libxgboost.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

java/libxgboost4j.so: java/xgboost4j_wrapper.cpp $(ALL_DEP)
	$(CXX) $(CFLAGS) $(JAVAINCFLAGS) -shared -o $@ $(filter %.cpp %.o %.a, $^) $(LDFLAGS)

xgboost: $(CLI_OBJ) $(ALL_DEP)
	$(CXX) $(CFLAGS) -o $@  $(filter %.o %.a, $^)  $(LDFLAGS)

rcpplint:
	python2 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} R-package/src

lint: rcpplint
	python2 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} include src plugin

clean:
	$(RM) -rf build build_plugin lib bin *~ */*~ */*/*~ */*/*/*~ amalgamation/*.o xgboost

clean_all: clean
	cd $(DMLC_CORE); make clean; cd -
	cd $(RABIT); make clean; cd -

# Script to make a clean installable R package.
Rpack:
	make clean_all
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
	make Rpack
	R CMD build --no-build-vignettes xgboost
	rm -rf xgboost

Rcheck:
	make Rbuild
	R CMD check  xgboost*.tar.gz

-include build/*.d
-include build/*/*.d
-include build_plugin/*/*.d
