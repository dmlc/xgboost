# specify tensor path
BIN = linear.rabit
MOCKBIN= 
MPIBIN = 
# objectives that makes up rabit library
OBJ = linear.o

# common build script for programs
include ../common.mk
CFLAGS+=-fopenmp
linear.o: linear.cc ../../src/*.h linear.h ../solver/*.h
# dependenies here
linear.rabit: linear.o lib
