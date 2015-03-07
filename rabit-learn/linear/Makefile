# specify tensor path
BIN = linear.rabit
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
