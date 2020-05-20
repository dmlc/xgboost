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

include $(DMLC_CORE)/make/dmlc.mk

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

export CFLAGS= -DDMLC_LOG_CUSTOMIZE=1 -std=c++14 -Wall -Wno-unknown-pragmas -Iinclude $(ADD_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include -I$(RABIT)/include -I$(GTEST_PATH)/include

ifeq ($(TEST_COVER), 1)
	CFLAGS += -g -O0 -fprofile-arcs -ftest-coverage
else
	CFLAGS += -O3 -funroll-loops
endif

ifndef LINT_LANG
	LINT_LANG= "all"
endif

# specify tensor path
.PHONY: clean all lint clean_all doxygen rcpplint pypack Rpack Rbuild Rcheck

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

# The should be equivalent to $(ALL_OBJ)  except for build/cli_main.o
amalgamation/xgboost-all0.o: amalgamation/xgboost-all0.cc
	$(CXX) -c $(CFLAGS) $< -o $@

rcpplint:
	python3 dmlc-core/scripts/lint.py xgboost ${LINT_LANG} R-package/src

lint: rcpplint
	python3 dmlc-core/scripts/lint.py --exclude_path python-package/xgboost/dmlc-core \
	  python-package/xgboost/include python-package/xgboost/lib \
	  python-package/xgboost/make python-package/xgboost/rabit \
	  python-package/xgboost/src --pylint-rc ${PWD}/python-package/.pylintrc xgboost \
	  ${LINT_LANG} include src python-package

ifeq ($(TEST_COVER), 1)
cover: check
	@- $(foreach COV_OBJ, $(COVER_OBJ), \
		gcov -pbcul -o $(shell dirname $(COV_OBJ)) $(COV_OBJ) > gcov.log || cat gcov.log; \
	)
endif

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o #xgboost
	$(RM) -rf build_tests *.gcov tests/cpp/xgboost_test
	if [ -d "R-package/src" ]; then \
		cd R-package/src; \
		$(RM) -rf rabit src include dmlc-core amalgamation *.so *.dll; \
		cd $(ROOTDIR); \
	fi

clean_all: clean
	cd $(DMLC_CORE); "$(MAKE)" clean; cd $(ROOTDIR)
	cd $(RABIT); "$(MAKE)" clean; cd $(ROOTDIR)

# create pip source dist (sdist) pack for PyPI
pippack: clean_all
	cd python-package; python setup.py sdist; mv dist/*.tar.gz ..; cd ..

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
# Modify PKGROOT in Makevars.in
	cat R-package/src/Makevars.in|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars.in
# Configure Makevars.win (Windows-specific Makevars, likely using MinGW)
	cp xgboost/src/Makevars.in xgboost/src/Makevars.win
	cat xgboost/src/Makevars.in| sed '3s/.*/ENABLE_STD_THREAD=0/' > xgboost/src/Makevars.win
	sed -i -e 's/@OPENMP_CXXFLAGS@/$$\(SHLIB_OPENMP_CXXFLAGS\)/g' xgboost/src/Makevars.win
	sed -i -e 's/-pthread/$$\(SHLIB_PTHREAD_FLAGS\)/g' xgboost/src/Makevars.win
	sed -i -e 's/@ENDIAN_FLAG@/-DDMLC_CMAKE_LITTLE_ENDIAN=1/g' xgboost/src/Makevars.win
	sed -i -e 's/@BACKTRACE_LIB@//g' xgboost/src/Makevars.win
	sed -i -e 's/@OPENMP_LIB@//g' xgboost/src/Makevars.win
	rm -f xgboost/src/Makevars.win-e   # OSX sed create this extra file; remove it
	bash R-package/remove_warning_suppression_pragma.sh
	rm xgboost/remove_warning_suppression_pragma.sh

Rbuild: Rpack
	R CMD build --no-build-vignettes xgboost
	rm -rf xgboost

Rcheck: Rbuild
	R CMD check xgboost*.tar.gz

-include build/*.d
-include build/*/*.d
