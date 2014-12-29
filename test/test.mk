ifndef $(nslave)
	nslave=2
endif
ifndef $(ndata)
	ndata=10
endif

# this is a makefile used to show testcases of rabit
.PHONY: model_recover local_recover speed


local_recover:
	../tracker/rabit_mpi.py -n $(nslave)  test_local_recover $(ndata) rabit_local_replica=1

local_recover_10_10k:
	../tracker/rabit_mpi.py -n 10  test_local_recover 10000 rabit_local_replica=1

# this experiment test recovery with actually process exit, use keepalive to keep program alive
model_recover_10_10k:
	../tracker/rabit_mpi.py -n 10 keepalive.sh test_model_recover 10000 mock=0,0,1,0 mock=1,1,1,0

model_recover_10_10k_die_same:
	../tracker/rabit_mpi.py -n 10 keepalive.sh test_model_recover 10000 mock=0,0,1,0 mock=1,1,1,0 mock=0,1,1,0 mock=4,1,1,0 mock=9,1,1,0

model_recover_10_10k_die_hard:
	../tracker/rabit_mpi.py -n 10 keepalive.sh test_model_recover 10000 mock=0,0,1,0 mock=1,1,1,0 mock=1,1,1,1 mock=0,1,1,0 mock=4,1,1,0 mock=9,1,1,0 mock=8,1,2,0 mock=4,1,3,0
