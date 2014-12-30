#!/usr/bin/python
"""
This is the demo submission script of rabit, it is created to
submit rabit jobs using hadoop streaming
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
#  Note: this submit script is only used for demo purpose
#  It does not have to be mpirun, it can be any job submission
#  script that starts the job, qsub, hadoop streaming etc.
#
def mpi_submit(nslave, slave_args):
    """
      customized submit script, that submit nslave jobs, each must contain args as parameter
      note this can be a lambda function containing additional parameters in input
      Parameters
         nslave number of slave process to start up
         args arguments to launch each job
              this usually includes the parameters of master_uri and parameters passed into submit
    """
    sargs = ' '.join(args.command + slave_args)
    if args.hostfile is None:
        cmd = ' '.join(['mpirun -n %d' % (nslave)] + args.command + slave_args) 
    else:
        ' '.join(['mpirun -n %d --hostfile %s' % (nslave, args.hostfile)] + args.command + slave_args)
    print cmd
    subprocess.check_call(cmd, shell = True)

# call submit, with nslave, the commands to run each job and submit function
tracker.submit(args.nworker, [], fun_submit = mpi_submit, verbose = args.verbose)
