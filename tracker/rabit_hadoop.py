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

#!!! Set path to hadoop and hadoop streaming jar here
hadoop_binary = 'hadoop'
#hadoop_streaming_jar = None
hadoop_streaming_jar = '/home/likewise-open/APEXLAB/blchen/streaming.jar'

# code 
hadoop_home = os.getenv('HADOOP_HOME')
if hadoop_home != None:
    if hadoop_binary == None:
        hadoop_binary = hadoop_home + '/bin/hadoop'
        assert os.path.exists(hadoop_binary), "HADDOP_HOME does not contain the hadoop binary"
    if hadoop_streaming_jar == None:
        hadoop_streaming_jar = hadoop_home + '/lib/hadoop-streaming.jar'
        assert os.path.exists(hadoop_streaming_jar), "HADDOP_HOME does not contain the haddop streaming jar"

if hadoop_binary == None or hadoop_streaming_jar == None:
    print 'Warning: Cannot auto-detect path to hadoop and hadoop-streaming jar, need to set them via arguments -hs and -hb'
    print '\tTo enable auto-detection, you can set enviroment variable HADOOP_HOME or modify rabit_hadoop.py line 14'

parser = argparse.ArgumentParser(description='Rabit script to submit rabit jobs using Hadoop Streaming')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-i', '--input', required=True,
                    help = 'input path in HDFS')
parser.add_argument('-o', '--output', required=True,
                    help = 'output path in HDFS')
parser.add_argument('-v', '--verbose', default=0, choices=[0, 1], type=int,
                    help = 'print more messages into the console')
parser.add_argument('-ac', '--auto_file_cache', default=1, choices=[0, 1], type=int,
                    help = 'whether automatically cache the files in the command to hadoop localfile, this is on by default')
parser.add_argument('-f', '--files', nargs = '*',
                    help = 'the cached file list in mapreduce,'\
                        ' the submission script will automatically cache all the files which appears in command to local folder'\
                        ' This will also cause rewritten of all the file names in the command to current path,'\
                        ' for example `../../kmeans ../kmeans.conf` will be rewritten to `./kmeans kmeans.conf`'\
                        ' because the two files are cached to running folder.'\
                        ' You may need this option to cache additional files.'\
                        ' You can also use it to manually cache files when auto_file_cache is off')
parser.add_argument('--jobname', default='auto', help = 'customize jobname in tracker')
parser.add_argument('--timeout', default=600000000, type=int,
                    help = 'timeout ((in milli seconds)) of each mapper job, automatically set to a very long time,'\
                        'normally you do not need to set this ')
parser.add_argument('-m', '--memory_mb', default=-1, type=int,
                    help = 'maximum memory used by the process, Guide: set it large (near mapred.cluster.max.map.memory.mb)'\
                        'if you are running multi-threading rabit,'\
                        'so that each node can occupy all the mapper slots in a machine for maximum performance')
if hadoop_binary == None:
    parser.add_argument('-hb', '--hadoop_binary', required = True,
                        help="path-to-hadoop binary folder")  
else:
    parser.add_argument('-hb', '--hadoop_binary', default = hadoop_binary, 
                        help="path-to-hadoop binary folder")  

if hadoop_streaming_jar == None:
    parser.add_argument('-hs', '--hadoop_streaming_jar', required = True,
                        help='path-to hadoop streamimg jar file')
else:
    parser.add_argument('-hs', '--hadoop_streaming_jar', default = hadoop_streaming_jar,
                        help='path-to hadoop streamimg jar file')
parser.add_argument('command', nargs='+',
                    help = 'command for rabit program')
args = parser.parse_args()

if args.jobname == 'auto':
    args.jobname = ('Rabit[nworker=%d]:' % args.nworker) + args.command[0].split('/')[-1];

def hadoop_streaming(nworker, worker_args):
    fset = set()
    if args.auto_file_cache:
        for i in range(len(args.command)):
            f = args.command[i]
            if os.path.exists(f):
                fset.add(f)
                if i == 0:
                    args.command[i] = './' + args.command[i].split('/')[-1]                    
                else:
                    args.command[i] = args.command[i].split('/')[-1]    
    cmd = '%s jar %s -D mapred.map.tasks=%d' % (args.hadoop_binary, args.hadoop_streaming_jar, nworker)
    cmd += ' -Dmapred.job.name=%s' % (args.jobname)
    cmd += ' -Dmapred.task.timeout=%d' % (args.timeout)
    cmd += ' -Dmapred.job.map.memory.mb=%d' % (args.memory_mb)
    cmd += ' -input %s -output %s' % (args.input, args.output)
    cmd += ' -mapper \"%s\" -reducer \"/bin/cat\" ' % (' '.join(args.command + worker_args))
    if args.files != None:
        for flst in args.files:
            for f in flst.split('#'):
                fset.add(f)
    for f in fset:
        cmd += ' -file %s' % f
    print cmd
    subprocess.check_call(cmd, shell = True)

tracker.submit(args.nworker, [], fun_submit = hadoop_streaming, verbose = args.verbose)
