#!/usr/bin/env python
"""
this script helps setup classpath env for HDFS, before running program
that links with libhdfs
"""
import glob
import sys
import os
import subprocess

if len(sys.argv) < 2:
    print 'Usage: the command you want to run'

hadoop_home = os.getenv('HADOOP_HOME')
hdfs_home = os.getenv('HADOOP_HDFS_HOME')
java_home = os.getenv('JAVA_HOME')
if hadoop_home is None:
    hadoop_home = os.getenv('HADOOP_PREFIX')
assert hadoop_home is not None, 'need to set HADOOP_HOME'
assert hdfs_home is not None, 'need to set HADOOP_HDFS_HOME'
assert java_home is not None, 'need to set JAVA_HOME'

(classpath, err) = subprocess.Popen('%s/bin/hadoop classpath' % hadoop_home,
                                    stdout=subprocess.PIPE, shell = True,
                                    env = os.environ).communicate()
cpath = []
for f in classpath.split(':'):
    cpath += glob.glob(f)

lpath = []
lpath.append('%s/lib/native' % hdfs_home)
lpath.append('%s/jre/lib/amd64/server' % java_home) 

env = os.environ.copy()
env['CLASSPATH'] = '${CLASSPATH}:' + (':'.join(cpath))

# setup hdfs options
if 'rabit_hdfs_opts' in env:
    env['LIBHDFS_OPTS'] = env['rabit_hdfs_opts']
elif 'LIBHDFS_OPTS' not in env:
    env['LIBHDFS_OPTS'] = '--Xmx128m'

env['LD_LIBRARY_PATH'] = '${LD_LIBRARY_PATH}:' + (':'.join(lpath)) 
ret = subprocess.call(args = sys.argv[1:], env = env)
sys.exit(ret)
