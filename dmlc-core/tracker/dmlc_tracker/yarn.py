"""
This is a script to submit dmlc job via Yarn
dmlc will run as a Yarn application
"""
# pylint: disable=invalid-name, too-many-locals, too-many-branches, missing-docstring
from __future__ import absolute_import
import os
import subprocess
import warnings
import logging
import platform
from threading import Thread
from . import opts
from . import tracker

def yarn_submit(args, nworker, nserver, pass_env):
    """Submission function for YARN."""
    is_windows = os.name == 'nt'
    hadoop_home = os.getenv('HADOOP_HOME')
    assert hadoop_home is not None, 'Need to set HADOOP_HOME for YARN submission.'
    hadoop_binary = os.path.join(hadoop_home, 'bin', 'hadoop')
    assert os.path.exists(hadoop_binary), "HADOOP_HOME does not contain the hadoop binary"

    if args.jobname is None:
        if args.num_servers == 0:
            prefix = ('DMLC[nworker=%d]:' % args.num_workers)
        else:
            prefix = ('DMLC[nworker=%d,nsever=%d]:' % (args.num_workers, args.num_servers))
        args.jobname = prefix + args.command[0].split('/')[-1]

    # Determine path for Yarn helpers
    YARN_JAR_PATH = os.path.join(args.yarn_app_dir, 'dmlc-yarn.jar')
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    YARN_BOOT_PY = os.path.join(curr_path, 'launcher.py')

    if not os.path.exists(YARN_JAR_PATH):
        warnings.warn("cannot find \"%s\", I will try to run build" % YARN_JAR_PATH)
        cmd = 'cd %s;./build.%s' % \
              (os.path.join(os.path.dirname(__file__), os.pardir, 'yarn'),
               'bat' if is_windows else 'sh')
        print(cmd)
        subprocess.check_call(cmd, shell=True, env=os.environ)
        assert os.path.exists(YARN_JAR_PATH), "failed to build dmlc-yarn.jar, try it manually"

    # detech hadoop version
    (out, _) = subprocess.Popen('%s version' % hadoop_binary,
                                shell=True, stdout=subprocess.PIPE).communicate()
    out = out.split('\n')[0].split()
    assert out[0] == 'Hadoop', 'cannot parse hadoop version string'
    hadoop_version = int(out[1].split('.')[0])
    (classpath, _) = subprocess.Popen('%s classpath' % hadoop_binary,
                                      shell=True, stdout=subprocess.PIPE).communicate()
    classpath = classpath.strip()

    if hadoop_version < 2:
        raise RuntimeError('Hadoop Version is %s, dmlc_yarn will need Yarn(Hadoop 2.0)' % out[1])

    fset, new_command = opts.get_cache_file_set(args)
    fset.add(YARN_JAR_PATH)
    fset.add(YARN_BOOT_PY)
    ar_list = []

    for fname in args.archives:
        fset.add(fname)
        ar_list.append(os.path.basename(fname))

    JAVA_HOME = os.getenv('JAVA_HOME')
    if JAVA_HOME is None:
        JAVA = 'java'
    else:
        JAVA = os.path.join(JAVA_HOME, 'bin', 'java')
    cmd = '%s -cp %s%s%s org.apache.hadoop.yarn.dmlc.Client '\
          % (JAVA, classpath, ';' if is_windows else ':', YARN_JAR_PATH)
    env = os.environ.copy()
    for k, v in pass_env.items():
        env[k] = str(v)

    # ship lib-stdc++.so
    if args.ship_libcxx is not None:
        if platform.architecture()[0] == '64bit':
            libcxx = args.ship_libcxx + '/libstdc++.so.6'
        else:
            libcxx = args.ship_libcxx + '/libstdc++.so'
        fset.add(libcxx)
        # update local LD_LIBRARY_PATH
        LD_LIBRARY_PATH = env['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in env else ''
        env['LD_LIBRARY_PATH'] = args.ship_libcxx + ':' + LD_LIBRARY_PATH

    env['DMLC_JOB_CLUSTER'] = 'yarn'
    env['DMLC_WORKER_CORES'] = str(args.worker_cores)
    env['DMLC_WORKER_MEMORY_MB'] = str(args.worker_memory_mb)
    env['DMLC_SERVER_CORES'] = str(args.server_cores)
    env['DMLC_SERVER_MEMORY_MB'] = str(args.server_memory_mb)
    env['DMLC_NUM_WORKER'] = str(args.num_workers)
    env['DMLC_NUM_SERVER'] = str(args.num_servers)
    env['DMLC_JOB_ARCHIVES'] = ':'.join(ar_list)

    for f in fset:
        cmd += ' -file %s' % f
    cmd += ' -jobname %s ' % args.jobname
    cmd += ' -tempdir %s ' % args.hdfs_tempdir
    cmd += ' -queue %s ' % args.queue
    if args.yarn_app_classpath:
        cmd += ' -appcp %s ' % args.yarn_app_classpath
    for entry in args.env:
        cmd += ' -env %s ' % entry
    cmd += (' '.join(['./launcher.py'] + new_command))

    logging.debug("Submit job with %d workers and %d servers", nworker, nserver)
    def run():
        """internal running function."""
        logging.debug(cmd)
        subprocess.check_call(cmd, shell=True, env=env)

    thread = Thread(target=run, args=())
    thread.setDaemon(True)
    thread.start()
    return thread

def submit(args):
    submit_thread = []
    def yarn_submit_pass(nworker, nserver, pass_env):
        submit_thread.append(yarn_submit(args, nworker, nserver, pass_env))

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    YARN_BOOT_PY = os.path.join(curr_path, 'launcher.py')
    tracker.submit(args.num_workers, args.num_servers,
                   fun_submit=yarn_submit_pass,
                   pscmd=(' '.join([YARN_BOOT_PY] + args.command)))
    for thread in submit_thread:
        thread.join()
