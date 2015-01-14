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
WRAPPER_PATH = os.path.dirname(__file__) + '/../wrapper'

parser = argparse.ArgumentParser(description='Rabit script to submit rabit job locally using python subprocess')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-v', '--verbose', default=0, choices=[0, 1], type=int,
                    help = 'print more messages into the console')
parser.add_argument('command', nargs='+',
                    help = 'command for rabit program')
args = parser.parse_args()

def exec_cmd(cmd, taskid):
    if cmd[0].find('/') == -1 and os.path.exists(cmd[0]):
        cmd[0] = './' + cmd[0]
    cmd = ' '.join(cmd)
    ntrial = 0
    while True:
        prep = 'PYTHONPATH=\"%s\" ' % WRAPPER_PATH
        arg = ' rabit_task_id=%d rabit_num_trial=%d' % (taskid, ntrial)        
        ret = subprocess.call(prep + cmd + arg, shell = True)
        if ret == 254 or ret == -2:
            ntrial += 1
            continue
        if ret == 0:
            return
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
