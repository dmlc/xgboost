#!/usr/bin/python
"""
This is an example job submit script for hadoop streaming
"""
import argparse
import sys
import os
import subprocess
sys.path.append('./src/')
import rabit_master as master

parser = argparse.ArgumentParser(description='Hadoop Streaming submission script')
parser.add_argument('-s', '--nslaves', required=True, type=int)
parser.add_argument('-hb', '--hadoop_binary', required=True)
parser.add_argument('-hs', '--hadoop_streaming_jar', required=True)
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-m', '--mapper', required=True)
#parser.add_argument('-r', '--reducer', required=False)
parser.add_argument('-k', '--nclusters', required=True, type=int)
parser.add_argument('-itr', '--iterations', required=True, type=int)
args = parser.parse_args()

def hadoop_streaming(nslaves, slave_args):
  cmd = '%s jar %s -input %s -output %s -mapper \"%s stdin %d %d stdout %s\" -reducer \"/bin/cat\" -file %s' % (args.hadoop_binary, args.hadoop_streaming_jar, args.input, args.output, args.mapper, args.nclusters, args.iterations, ' '.join(slave_args), args.mapper)
  print cmd
  subprocess.check_call(cmd, shell = True)
  
master.submit(args.nslaves, [], fun_submit= hadoop_streaming)
