#!/usr/bin/python
"""
This is a script to submit rabit job using hadoop streaming
submit the rabit process as mappers of MapReduce
"""
import argparse
import sys
import os
import time
import subprocess
import rabit_tracker as tracker

#!!! you can directly set hadoop binary path and hadoop streaming path here
hadoop_binary = 'hadoop'
hadoop_streaming_jar = None

parser = argparse.ArgumentParser(description='Rabit script to submit rabit jobs using hadoop streaming')
parser.add_argument('-s', '--nslaves', required=True, type=int,
                    help = "number of slaves proccess to be launched")
if hadoop_binary == None:
  parser.add_argument('-hb', '--hadoop_binary', required=True,
                      help="path-to-hadoop binary folder")
if hadoop_streaming_jar == None:
  parser.add_argument('-hs', '--hadoop_streaming_jar', required=True,
                      help='path-to hadoop streamimg jar file')
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-m', '--mapper', required=True)
parser.add_argument('-a', '--args', required=True)
args = parser.parse_args()

if hadoop_binary != None:
  args.hadoop_binary = hadoop_binary
if hadoop_streaming_jar != None:
  args.hadoop_streaming_jar = hadoop_streaming_jar

def hadoop_streaming(nslaves, slave_args):
  cmd = '%s jar %s' % (args.hadoop_binary, args.hadoop_streaming_jar)
  cmd += ' -input %s -output %s' % (args.input, args.output)
  cmd += ' -mapper \"%s %s %s\" -reducer \"/bin/cat\" ' % (args.mapper, args.args, ' '.join(slave_args))
  cmd += ' -file %s -D mapred.map.tasks=%d' % (args.mapper, nslaves)
  print cmd
  subprocess.check_call(cmd, shell = True)

start = time.time()
tracker.submit(args.nslaves, [], fun_submit= hadoop_streaming)
print 'All run took %s' % (time.time() - start)
