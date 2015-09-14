export CC  = gcc
#build on the fly
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops
# java include path
export JAVAINCFLAGS = -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -I./java

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
	#CFLAGS += -fopenmp
	ifeq ($(omp_mac_static),1)
		#CFLAGS += -fopenmp -Bstatic
		CFLAGS += -static-libgcc -static-libstdc++ -L. -fopenmp
		#LDFLAGS += -Wl,--whole-archive -lpthread -Wl --no-whole-archive
	else
		CFLAGS += -fopenmp
	endif
endif


# by default use c++11
ifeq ($(cxx11),1)
	CFLAGS += -std=c++11
endif

# handling dmlc
ifdef dmlc
	ifndef config
		ifneq ("$(wildcard $(dmlc)/config.mk)","")
			config = $(dmlc)/config.mk
		else
			config = $(dmlc)/make/config.mk
		endif
	endif
	include $(config)
	include $(dmlc)/make/dmlc.mk
	LDFLAGS+= $(DMLC_LDFLAGS)
	LIBDMLC=$(dmlc)/libdmlc.a
else
	LIBDMLC=dmlc_simple.o
endif

ifndef WITH_FPIC
	WITH_FPIC = 1
endif
ifeq ($(WITH_FPIC), 1)
	CFLAGS += -fPIC
endif


ifeq ($(OS), Windows_NT)
	LIBRABIT = subtree/rabit/lib/librabit_empty.a
	SLIB = wrapper/xgboost_wrapper.dll
else
	LIBRABIT = subtree/rabit/lib/librabit.a
	SLIB = wrapper/libxgboostwrapper.so
endif

# java lib
JLIB = java/libxgboostjavawrapper.so

# specify tensor path
BIN = xgboost
MOCKBIN = xgboost.mock
OBJ = updater.o gbm.o io.o main.o dmlc_simple.o
MPIBIN =
ifeq ($(WITH_FPIC), 1)
	TARGET = $(BIN) $(OBJ) $(SLIB)
else
	TARGET = $(BIN)
endif

ifndef LINT_LANG
	LINT_LANG= "all"
endif

.PHONY: clean all mpi python Rpack lint

all: $(TARGET)
mpi: $(MPIBIN)

python: wrapper/libxgboostwrapper.so
# now the wrapper takes in two files. io and wrapper part
updater.o: src/tree/updater.cpp  src/tree/*.hpp src/*.h src/tree/*.h src/utils/*.h
dmlc_simple.o: src/io/dmlc_simple.cpp src/utils/*.h
gbm.o: src/gbm/gbm.cpp src/gbm/*.hpp src/gbm/*.h
io.o: src/io/io.cpp src/io/*.hpp src/utils/*.h src/learner/dmatrix.h src/*.h
main.o: src/xgboost_main.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h
xgboost:  updater.o gbm.o io.o main.o $(LIBRABIT) $(LIBDMLC)
wrapper/xgboost_wrapper.dll wrapper/libxgboostwrapper.so: wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h  updater.o gbm.o io.o $(LIBRABIT) $(LIBDMLC)

java: java/libxgboostjavawrapper.so
java/libxgboostjavawrapper.so: java/xgboost4j_wrapper.cpp wrapper/xgboost_wrapper.cpp src/utils/*.h src/*.h src/learner/*.hpp src/learner/*.h  updater.o gbm.o io.o $(LIBRABIT) $(LIBDMLC)

# dependency on rabit
subtree/rabit/lib/librabit.a: subtree/rabit/src/engine.cc
	+	cd subtree/rabit;make lib/librabit.a; cd ../..
subtree/rabit/lib/librabit_empty.a: subtree/rabit/src/engine_empty.cc
	+	cd subtree/rabit;make lib/librabit_empty.a; cd ../..
subtree/rabit/lib/librabit_mock.a: subtree/rabit/src/engine_mock.cc
	+	cd subtree/rabit;make lib/librabit_mock.a; cd ../..
subtree/rabit/lib/librabit_mpi.a: subtree/rabit/src/engine_mpi.cc
	+	cd subtree/rabit;make lib/librabit_mpi.a; cd ../..

$(BIN) :
	$(CXX) $(CFLAGS) -fPIC -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS)

$(MOCKBIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS)

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS) $(DLLFLAGS)

$(JLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)  $(JAVAINCFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIOBJ) :
	$(MPICXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(MPIBIN) :
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^) $(LDFLAGS)

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

Rpack:
	make clean
	cd subtree/rabit;make clean;cd ..
	rm -rf xgboost xgboost*.tar.gz
	cp -r R-package xgboost
	rm -rf xgboost/src/*.o xgboost/src/*.so xgboost/src/*.dll
	rm -rf xgboost/src/*/*.o
	rm -rf subtree/rabit/src/*.o
	rm -rf xgboost/demo/*.model xgboost/demo/*.buffer xgboost/demo/*.txt
	rm -rf xgboost/demo/runall.R
	cp -r src xgboost/src/src
	mkdir xgboost/src/subtree
	mkdir xgboost/src/subtree/rabit
	cp -r subtree/rabit/include xgboost/src/subtree/rabit/include
	cp -r subtree/rabit/src xgboost/src/subtree/rabit/src
	rm -rf xgboost/src/subtree/rabit/src/*.o
	mkdir xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.h xgboost/src/wrapper
	cp  wrapper/xgboost_wrapper.cpp xgboost/src/wrapper
	cp ./LICENSE xgboost
	cat R-package/src/Makevars|sed '2s/.*/PKGROOT=./' > xgboost/src/Makevars
	cp xgboost/src/Makevars xgboost/src/Makevars.win
	# R CMD build --no-build-vignettes xgboost
	# R CMD build xgboost
	# rm -rf xgboost
	# R CMD check --as-cran xgboost*.tar.gz

Rbuild:
	make Rpack
	R CMD build xgboost
	rm -rf xgboost

Rcheck:
	make Rbuild
	R CMD check --as-cran xgboost*.tar.gz

pythonpack:
	#make clean
	cd subtree/rabit;make clean;cd ..
	rm -rf xgboost-deploy xgboost*.tar.gz
	cp -r python-package xgboost-deploy
	cp *.md xgboost-deploy/
	cp LICENSE xgboost-deploy/
	cp Makefile xgboost-deploy/xgboost
	cp -r wrapper xgboost-deploy/xgboost
	cp -r subtree xgboost-deploy/xgboost
	cp -r multi-node xgboost-deploy/xgboost
	cp -r windows xgboost-deploy/xgboost
	cp -r src xgboost-deploy/xgboost

	#make python

pythonbuild:
	make pythonpack
	python setup.py install

pythoncheck:
	make pythonbuild
	python -c 'import xgboost;print xgboost.core.find_lib_path()'

# lint requires dmlc to be in current folder
lint:
	dmlc-core/scripts/lint.py xgboost $(LINT_LANG) src wrapper R-package python-package

clean:
	$(RM) -rf $(OBJ) $(BIN) $(MPIBIN) $(MPIOBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
	cd subtree/rabit; make clean; cd ..
