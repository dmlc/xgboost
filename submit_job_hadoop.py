#!/usr/bin/python
"""
This is an example job submit script for hadoop streaming
"""
import argparse
import sys
import os
import subprocess
sys.path.append('./src/')
from rabit_master import Master
from threading import Thread


def hadoop_streaming(nslaves, slave_args):
  cmd = '%s jar %s -input %s -output %s -mapper \"%s %s\" -reducer /bin/cat stdin %d %d stdout' % (args.hadoop_binary, args.hadoop_streaming_jar, args.input, args.output, args.mapper, ' '.join(slave_args), args.nclusters, args.iterations)
  print cmd
  subprocess.check_call(cmd, shell = True)


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

master = Master()
# this is availabe after triggered the hadoop streaming job, not sure how to do it
# os.environ["mapred_job_id"]
slave_args = ['master_uri=%s' % 'TODO', 'master_port=%s' % 'TODO']
submit_thread = Thread(target = hadoop_streaming, args = slave_args)
submit_thread.start()
master.accept_slaves(args.nslaves)
submit_thread.join()
