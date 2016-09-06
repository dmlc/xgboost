#!/bin/bash
make -f test.mk model_recover_10_10k || exit -1
make -f test.mk model_recover_10_10k_die_same  || exit -1
make -f test.mk local_recover_10_10k || exit -1
make -f test.mk lazy_recover_10_10k_die_hard || exit -1
make -f test.mk lazy_recover_10_10k_die_same || exit -1
make -f test.mk ringallreduce_10_10k || exit -1
