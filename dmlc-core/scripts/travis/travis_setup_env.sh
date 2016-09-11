# script to be sourced in travis yml
# setup all enviroment variables

export PATH=${HOME}/.local/bin:${PATH}
export PATH=${PATH}:${CACHE_PREFIX}/bin
export CACHE_PREFIX=${HOME}/.cache/usr
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CACHE_PREFIX}/include
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${CACHE_PREFIX}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${CACHE_PREFIX}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CACHE_PREFIX}/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${CACHE_PREFIX}/lib

alias make="make -j4"

# setup the cache prefix folder
if [ ! -d ${HOME}/.cache ]; then
    mkdir ${HOME}/.cache
fi

if [ ! -d ${CACHE_PREFIX} ]; then
    mkdir ${CACHE_PREFIX}
fi
if [ ! -d ${CACHE_PREFIX}/include ]; then
    mkdir ${CACHE_PREFIX}/include
fi
if [ ! -d ${CACHE_PREFIX}/lib ]; then
    mkdir ${CACHE_PREFIX}/lib
fi
if [ ! -d ${CACHE_PREFIX}/bin ]; then
    mkdir ${CACHE_PREFIX}/bin
fi

# setup CUDA path if NVCC_PREFIX exists
if [ ! -z "$NVCC_PREFIX" ]; then
    export PATH=${PATH}:${NVCC_PREFIX}/usr/local/cuda-7.5/bin
    export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${NVCC_PREFIX}/usr/local/cuda-7.5/include
    export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${NVCC_PREFIX}/usr/local/cuda-7.5/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${NVCC_PREFIX}/usr/local/cuda-7.5/lib64:${NVCC_PREFIX}/usr/lib/x86_64-linux-gnu
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVCC_PREFIX}/usr/local/cuda-7.5/lib64:${NVCC_PREFIX}/usr/lib/x86_64-linux-gnu
fi
