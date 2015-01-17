#!/usr/bin/python
"""
This is the demo submission script of rabit, it is created to
submit rabit jobs using hadoop streaming
"""
import argparse
import sys
import os
import subprocess
from threading import Thread
import rabit_tracker as tracker
if os.name == 'nt':
    WRAPPER_PATH = os.path.dirname(__file__) + '\\..\\wrapper'
else:
    WRAPPER_PATH = os.path.dirname(__file__) + '/../wrapper'

parser = argparse.ArgumentParser(description='Rabit script to submit rabit job locally using python subprocess')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-v', '--verbose', default=0, choices=[0, 1], type=int,
                    help = 'print more messages into the console')
parser.add_argument('command', nargs='+',
                    help = 'command for rabit program')
args = parser.parse_args()

# bash script for keepalive
# use it so that python do not need to communicate with subprocess
echo="echo %s rabit_num_trial=$nrep;"
keepalive = """
nrep=0
rc=254
while [ $rc -eq 254 ]; 
do
    %s
    %s %s rabit_num_trial=$nrep
    rc=$?;
    nrep=$((nrep+1));
done
"""

def exec_cmd(cmd, taskid):
    if cmd[0].find('/') == -1 and os.path.exists(cmd[0]) and os.name != 'nt':
        cmd[0] = './' + cmd[0]
    cmd = ' '.join(cmd)
    arg = ' rabit_task_id=%d' % (taskid) 
    cmd = cmd + arg
    ntrial = 0
    while True:
        if os.name == 'nt':
            prep = 'SET PYTHONPATH=\"%s\"\n' % WRAPPER_PATH
            ret = subprocess.call(prep + cmd + ('rabit_num_trial=%d' % ntrial),
                                  shell=True)
            if ret == 254:
                ntrial += 1
                continue
            
        else:
            prep = 'PYTHONPATH=\"%s\" ' % WRAPPER_PATH
            if args.verbose != 0:            
                bash = keepalive % (echo % cmd, prep, cmd)
            else:
                bash = keepalive % ('', prep, cmd)            
            ret = subprocess.call(bash, shell=True, executable='bash')
        if ret == 0:
            if args.verbose != 0:        
                print 'Thread %d exit with 0' % taskid
            return
        else:
            raise Exception('Get nonzero return code=%d' % ret)
#
#  Note: this submit script is only used for demo purpose
#  submission script using pyhton multi-threading
#
def mthread_submit(nslave, worker_args):
    """
      customized submit script, that submit nslave jobs, each must contain args as parameter
      note this can be a lambda function containing additional parameters in input
      Parameters
         nslave number of slave process to start up
         args arguments to launch each job
              this usually includes the parameters of master_uri and parameters passed into submit
    """       
    procs = {}
    for i in range(nslave):
        procs[i] = Thread(target = exec_cmd, args = (args.command + worker_args, i))
        procs[i].daemon = True
        procs[i].start()
    for i in range(nslave):
        procs[i].join()

# call submit, with nslave, the commands to run each job and submit function
tracker.submit(args.nworker, [], fun_submit = mthread_submit, verbose = args.verbose)
