include Makefile.config
export CC  = gcc
export CXX = g++
export MPICXX = mpicxx
export LDFLAGS= -pthread -lm 
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops
export ROOT = ..

ifeq ($(OS), Windows_NT)
	export CXX = g++ -m64
	export CC = gcc -m64
endif

ifeq ($(no_omp),1)
	CFLAGS += -DDISABLE_OPENMP 
else 
	CFLAGS += -fopenmp
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



JLIB = libxgboostjavawrapper.so

java: libxgboostjavawrapper.so
libxgboostjavawrapper.so: xgboost4j_wrapper.h xgboost4j_wrapper.cpp $(ROOT)/src/utils/*.h\
 $(ROOT)/src/*.h $(ROOT)/src/learner/*.hpp $(ROOT)/src/learner/*.h $(ROOT)/wrapper/xgboost_wrapper.cpp\
  $(ROOT)/updater.o $(ROOT)/gbm.o $(ROOT)/io.o $(ROOT)/$(LIBRABIT) $(ROOT)/$(LIBDMLC)

$(JLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.so %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)  $(JAVAINCFLAGS)