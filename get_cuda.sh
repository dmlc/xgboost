export CUDA_HOME=/usr/local/cuda ; if [ -d "${CUDA_HOME}" ]; then \
export CUDA_LIB=${CUDA_HOME}/lib64 && \
export CUDA_VERSION=`ls ${CUDA_LIB}/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev` && \
export CUDA_MAJOR=`echo ${CUDA_VERSION} | cut -d "." -f 1` && \
export CUDA_MINOR=`echo ${CUDA_VERSION} | cut -d "." -f 2| sed 's/@//g'` && \
export CUDA_SHORT=`echo ${CUDA_VERSION} | egrep -o '[0-9]\.[0-9]'` && \
echo ${CUDA_SHORT} > cuda_short.txt ; \
fi

export cuda=`cat cuda_short.txt`
export arch=`arch`
if [[ $cuda == '9.2' ]] && [[ $arch == 'ppc64le' ]] || [[ $arch == 'x86_64' ]]; then export NCCL=-DUSE_NCCL=ON ; else export NCCL= ; fi

echo ${NCCL}
