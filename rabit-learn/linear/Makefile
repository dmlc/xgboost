ifneq ("$(wildcard ../config.mk)","")
	config = ../config.mk
else
	config = ../make/config.mk
endif
include $(config)

BIN = linear.rabit test.rabit
MOCKBIN= linear.mock
MPIBIN = 
# objectives that makes up rabit library
OBJ = linear.o

# common build script for programs
include ../make/common.mk
CFLAGS+=-fopenmp
linear.o: linear.cc ../../src/*.h linear.h ../solver/*.h
# dependenies here
linear.rabit: linear.o lib
linear.mock: linear.o lib
test.rabit: test_load.cc lib
