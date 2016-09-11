"""Submit jobs to Sun Grid Engine."""
# pylint: disable=invalid-name
from __future__ import absolute_import

import os
import subprocess
from . import tracker

def submit(args):
    """Job submission script for SGE."""
    if args.jobname is None:
        args.jobname = ('dmlc%d.' % args.num_workers) + args.command[0].split('/')[-1]
    if args.sge_log_dir is None:
        args.sge_log_dir = args.jobname + '.log'

    if os.path.exists(args.sge_log_dir):
        if not os.path.isdir(args.sge_log_dir):
            raise RuntimeError('specified --sge-log-dir %s is not a dir' % args.sge_log_dir)
    else:
        os.mkdir(args.sge_log_dir)

    runscript = '%s/rundmlc.sh' % args.logdir
    fo = open(runscript, 'w')
    fo.write('source ~/.bashrc\n')
    fo.write('export DMLC_TASK_ID=${SGE_TASK_ID}\n')
    fo.write('export DMLC_JOB_CLUSTER=sge\n')
    fo.write('\"$@\"\n')
    fo.close()

    def sge_submit(nworker, nserver, pass_envs):
        """Internal submission function."""
        env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in pass_envs.items()])
        cmd = 'qsub -cwd -t 1-%d -S /bin/bash' % (nworker + nserver)
        if args.queue != 'default':
            cmd += '-q %s' % args.queue
        cmd += ' -N %s ' % args.jobname
        cmd += ' -e %s -o %s' % (args.logdir, args.logdir)
        cmd += ' -pe orte %d' % (args.vcores)
        cmd += ' -v %s,PATH=${PATH}:.' % env_arg
        cmd += ' %s %s' % (runscript, ' '.join(args.command))
        print(cmd)
        subprocess.check_call(cmd, shell=True)
        print('Waiting for the jobs to get up...')

    # call submit, with nslave, the commands to run each job and submit function
    tracker.submit(args.num_workers, args.num_servers,
                   fun_submit=sge_submit,
                   pscmd=' '.join(args.command))
