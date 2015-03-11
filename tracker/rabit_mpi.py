#!/usr/bin/env python
"""
Submission script to submit rabit jobs using MPI
"""
import argparse
import sys
import os
import subprocess
import rabit_tracker as tracker

parser = argparse.ArgumentParser(description='Rabit script to submit rabit job using MPI')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-v', '--verbose', default=0, choices=[0, 1], type=int,
                    help = 'print more messages into the console')
parser.add_argument('-H', '--hostfile', type=str,
                    help = 'the hostfile of mpi server')
parser.add_argument('command', nargs='+',
                    help = 'command for rabit program')
args = parser.parse_args()
#
# submission script using MPI
#
def mpi_submit(nslave, worker_args, worker_envs):
    """
      customized submit script, that submit nslave jobs, each must contain args as parameter
      note this can be a lambda function containing additional parameters in input
      Parameters
         nslave number of slave process to start up
         args arguments to launch each job
              this usually includes the parameters of master_uri and parameters passed into submit
    """
    worker_args += ['%s=%s' % (k, str(v)) for k, v in worker_envs.items()]
    sargs = ' '.join(args.command + worker_args)
    if args.hostfile is None:
        cmd = ' '.join(['mpirun -n %d' % (nslave)] + args.command + worker_args) 
    else:
        cmd = ' '.join(['mpirun -n %d --hostfile %s' % (nslave, args.hostfile)] + args.command + worker_args)
    print cmd
    subprocess.check_call(cmd, shell = True)

# call submit, with nslave, the commands to run each job and submit function
tracker.submit(args.nworker, [], fun_submit = mpi_submit, verbose = args.verbose)
