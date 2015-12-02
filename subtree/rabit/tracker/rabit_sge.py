#!/usr/bin/env python
"""
Submit rabit jobs to Sun Grid Engine
"""
import argparse
import sys
import os
import subprocess
import rabit_tracker as tracker

parser = argparse.ArgumentParser(description='Rabit script to submit rabit job using MPI')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-q', '--queue', default='default', type=str,
                    help = 'the queue we want to submit the job to')
parser.add_argument('-hip', '--host_ip', default='auto', type=str,
                    help = 'host IP address if cannot be automatically guessed, specify the IP of submission machine')
parser.add_argument('--vcores', default = 1, type=int,
                    help = 'number of vcpores to request in each mapper, set it if each rabit job is multi-threaded')
parser.add_argument('--jobname', default='auto', help = 'customize jobname in tracker')
parser.add_argument('--logdir', default='auto', help = 'customize the directory to place the logs')
parser.add_argument('-v', '--verbose', default=0, choices=[0, 1], type=int,
                    help = 'print more messages into the console')
parser.add_argument('command', nargs='+',
                    help = 'command for rabit program')
args = parser.parse_args()

if args.jobname == 'auto':
    args.jobname = ('rabit%d.' % args.nworker) + args.command[0].split('/')[-1];
if args.logdir == 'auto':
    args.logdir = args.jobname + '.log'

if os.path.exists(args.logdir):
    if not os.path.isdir(args.logdir):
        raise RuntimeError('specified logdir %s is a file instead of directory' % args.logdir)
else:
    os.mkdir(args.logdir)
    
runscript = '%s/runrabit.sh' % args.logdir
fo = open(runscript, 'w')
fo.write('source ~/.bashrc\n')
fo.write('\"$@\"\n')
fo.close()
#
# submission script using MPI
#
def sge_submit(nslave, worker_args, worker_envs):
    """
      customized submit script, that submit nslave jobs, each must contain args as parameter
      note this can be a lambda function containing additional parameters in input
      Parameters
         nslave number of slave process to start up
         args arguments to launch each job
              this usually includes the parameters of master_uri and parameters passed into submit
    """
    env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in worker_envs.items()])
    cmd = 'qsub -cwd -t 1-%d -S /bin/bash' % nslave
    if args.queue != 'default':
        cmd += '-q %s' % args.queue
    cmd += ' -N %s ' % args.jobname
    cmd += ' -e %s -o %s' % (args.logdir, args.logdir)
    cmd += ' -pe orte %d' % (args.vcores)
    cmd += ' -v %s,PATH=${PATH}:.' % env_arg
    cmd += ' %s %s' % (runscript, ' '.join(args.command + worker_args))
    print cmd
    subprocess.check_call(cmd, shell = True)
    print 'Waiting for the jobs to get up...'

# call submit, with nslave, the commands to run each job and submit function
tracker.submit(args.nworker, [], fun_submit = sge_submit, verbose = args.verbose)
