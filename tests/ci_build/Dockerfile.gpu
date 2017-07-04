FROM nvidia/cuda:8.0-devel-ubuntu14.04

RUN apt-get update && apt-get -y upgrade
# CMAKE
RUN sudo apt-get install -y build-essential
RUN apt-get install -y wget
RUN wget http://www.cmake.org/files/v3.5/cmake-3.5.2.tar.gz 
RUN tar -xvzf cmake-3.5.2.tar.gz 
RUN cd cmake-3.5.2/ && ./configure && make && sudo make install

# BLAS
RUN apt-get install -y libatlas-base-dev

# PYTHON2
RUN apt-get install -y python-setuptools python-pip python-dev unzip gfortran
RUN pip install numpy nose scipy scikit-learn
