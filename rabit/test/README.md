Testcases of Rabit
====
This folder contains internal testcases to test correctness and efficiency of rabit API

The example running scripts for testcases are given by test.mk
* type ```make -f test.mk testcasename``` to run certain testcase


Helper Scripts
====
* test.mk contains Makefile documentation of all testcases
* keepalive.sh helper bash to restart a program when it dies abnormally

List of Programs
====
* speed_test: test the running speed of rabit API
* test_local_recover: test recovery of local state when error happens
* test_model_recover: test recovery of global state when error happens
