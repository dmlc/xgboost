#!/bin/bash

make -f test.mk RABIT_BUILD_DMLC=1 model_recover_10_10k || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 model_recover_10_10k_die_same  || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 model_recover_10_10k_die_hard || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 local_recover_10_10k || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 lazy_recover_10_10k_die_hard || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 lazy_recover_10_10k_die_same || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 ringallreduce_10_10k || exit -1
make -f test.mk RABIT_BUILD_DMLC=1 pylocal_recover_10_10k || exit -1
