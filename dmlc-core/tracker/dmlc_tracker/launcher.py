#!/usr/bin/env python
# pylint: disable=invalid-name
"""The container launcher script that launches DMLC with the right env variable."""
import glob
import sys
import os
import subprocess

def unzip_archives(ar_list, env):
    for fname in ar_list:
        if not os.path.exists(fname):
            continue
        if fname.endswith('.zip'):
            subprocess.call(args=['unzip', fname], env=env)
        elif fname.find('.tar') != -1:
            subprocess.call(args=['tar', 'xf', fname], env=env)

def main():
    """Main moduke of the launcher."""
    if len(sys.argv) < 2:
        print('Usage: launcher.py your command')
        sys.exit(0)

    hadoop_home = os.getenv('HADOOP_HOME')
    hdfs_home = os.getenv('HADOOP_HDFS_HOME')
    java_home = os.getenv('JAVA_HOME')
    hadoop_home = os.getenv('HADOOP_PREFIX') if hadoop_home is None else hadoop_home
    cluster = os.getenv('DMLC_JOB_CLUSTER')

    assert cluster is not None, 'need to have DMLC_JOB_CLUSTER'

    env = os.environ.copy()
    library_path = ['./']
    class_path = []

    if cluster == 'yarn':
        assert hadoop_home is not None, 'need to set HADOOP_HOME'
        assert hdfs_home is not None, 'need to set HADOOP_HDFS_HOME'
        assert java_home is not None, 'need to set JAVA_HOME'

    if cluster == 'sge':
        num_worker = int(env['DMLC_NUM_WORKER'])
        task_id = int(env['DMLC_TASK_ID'])
        if task_id < num_worker:
            env['DMLC_ROLE'] = 'worker'
        else:
            env['DMLC_ROLE'] = 'server'

    if hadoop_home:
        library_path.append('%s/lib/native' % hdfs_home)
        library_path.append('%s/lib' % hdfs_home)
        (classpath, _) = subprocess.Popen('%s/bin/hadoop classpath' % hadoop_home,
                                          stdout=subprocess.PIPE, shell=True,
                                          env=os.environ).communicate()
        for f in classpath.split(':'):
            class_path += glob.glob(f)

    if java_home:
        library_path.append('%s/jre/lib/amd64/server' % java_home)

    env['CLASSPATH'] = '${CLASSPATH}:' + (':'.join(class_path))

    # setup hdfs options
    if 'DMLC_HDFS_OPTS' in env:
        env['LIBHDFS_OPTS'] = env['DMLC_HDFS_OPTS']
    elif 'LIBHDFS_OPTS' not in env:
        env['LIBHDFS_OPTS'] = '--Xmx128m'

    LD_LIBRARY_PATH = env['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in env else ''
    env['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH + ':' + ':'.join(library_path)

    # unzip the archives.
    if 'DMLC_JOB_ARCHIVES' in env:
        unzip_archives(env['DMLC_JOB_ARCHIVES'].split(':'), env)

    ret = subprocess.call(args=sys.argv[1:], env=env)
    sys.exit(ret)


if __name__ == '__main__':
    main()
