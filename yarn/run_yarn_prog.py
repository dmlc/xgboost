#!/usr/bin/env python
"""
this script helps setup classpath env for HDFS
"""
import glob
import sys
import os
import subprocess

if len(sys.argv) < 2:
    print 'Usage: the program you want to run'

hadoop_home = os.getenv('HADOOP_HOME')
if hadoop_home is None:
    hadoop_home = os.getenv('HADOOP_PREFIX')
assert hadoop_home is not None, 'need to set HADOOP_HOME'

(classpath, err) = subprocess.Popen('%s/bin/hadoop classpath' % hadoop_home, shell = True, stdout=subprocess.PIPE, env = os.environ).communicate()
cpath = []
for f in classpath.split(':'):
    cpath += glob.glob(f)

env = os.environ.copy()
env['CLASSPATH'] = '${CLASSPATH}:' + (':'.join(cpath))
subprocess.check_call(' '.join(sys.argv[1:]), shell = True, env = env)
