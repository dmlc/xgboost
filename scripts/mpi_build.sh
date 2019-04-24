#!/usr/bin/env bash

if [ -f mpich/lib/libmpich.so ]; then
  echo "libmpich.so found -- nothing to build."
else
  echo "Downloading mpich source."
  wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
  tar xfz mpich-3.2.tar.gz
  rm mpich-3.2.tar.gz*
  echo "configuring and building mpich."
  cd mpich-3.2
  #CC=gcc CXX=g++ CFLAGS=-m64 CXXFLAGS=-m64 FFLAGS=-m64
  ./configure \
          --prefix=`pwd`/../mpich \
          --enable-static=false \
          --enable-alloca=true \
          --disable-long-double \
          --enable-threads=single \
          --enable-fortran=no \
          --enable-fast=all \
          --enable-g=none \
          --enable-timing=none \
          --enable-cxx
  make -j4
  make install
  cd -
fi