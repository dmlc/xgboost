#!/usr/bin/python
"""
This is a script to submit rabit job using Yarn
submit the rabit process as mappers of MapReduce
"""
import rabit_hadoop

if __name__ == 'main':
    fun_submit = lambda nworker, worker_args: hadoop_streaming(nworker, worker_args, yarn=True)
    tracker.submit(args.nworker, [], fun_submit = fun_submit, verbose = args.verbose)
