# pylint: disable=invalid-name
"""Command line options of job submission script."""
import os
import argparse

def get_cache_file_set(args):
    """Get the list of files to be cached.

    Parameters
    ----------
    args: ArgumentParser.Argument
        The arguments returned by the parser.

    Returns
    -------
    cache_file_set: set of str
        The set of files to be cached to local execution environment.

    command: list of str
        The commands that get rewritten after the file cache is used.
    """
    fset = set()
    cmds = []
    if args.auto_file_cache:
        for i in range(len(args.command)):
            fname = args.command[i]
            if os.path.exists(fname):
                fset.add(fname)
                cmds.append('./' + fname.split('/')[-1])
            else:
                cmds.append(fname)

    for fname in args.files:
        if os.path.exists(fname):
            fset.add(fname)
    return fset, cmds


def get_memory_mb(mem_str):
    """Get the memory in MB from memory string.

    mem_str: str
        String representation of memory requirement.

    Returns
    -------
    mem_mb: int
        Memory requirement in MB.
    """
    mem_str = mem_str.lower()
    if mem_str.endswith('g'):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith('m'):
        return int(float(mem_str[:-1]))
    else:
        msg = 'Invalid memory specification %s, need to be a number follows g or m' % mem_str
        raise RuntimeError(msg)


def get_opts(args=None):
    """Get options to launch the job.

    Returns
    -------
    args: ArgumentParser.Argument
        The arguments returned by the parser.

    cache_file_set: set of str
        The set of files to be cached to local execution environment.
    """
    parser = argparse.ArgumentParser(description='DMLC job submission.')
    parser.add_argument('--cluster', type=str,
                        choices=['yarn', 'mpi', 'sge', 'local', 'ssh'],
                        help=('Cluster type of this submission,' +
                              'default to env variable ${DMLC_SUBMIT_CLUSTER}.'))
    parser.add_argument('--num-workers', required=True, type=int,
                        help='Number of worker proccess to be launched.')
    parser.add_argument('--worker-cores', default=1, type=int,
                        help='Number of cores to be allocated for each worker process.')
    parser.add_argument('--worker-memory', default='1g', type=str,
                        help=('Memory need to be allocated for each worker,' +
                              ' need to ends with g or m'))
    parser.add_argument('--num-servers', default=0, type=int,
                        help='Number of server process to be launched. Only used in PS jobs.')
    parser.add_argument('--server-cores', default=1, type=int,
                        help=('Number of cores to be allocated for each server process.' +
                              'Only used in PS jobs.'))
    parser.add_argument('--server-memory', default='1g', type=str,
                        help=('Memory need to be allocated for each server, ' +
                              'need to ends with g or m.'))
    parser.add_argument('--jobname', default=None, type=str, help='Name of the job.')
    parser.add_argument('--queue', default='default', type=str,
                        help='The submission queue the job should goes to.')
    parser.add_argument('--log-level', default='INFO', type=str,
                        choices=['INFO', 'DEBUG'],
                        help='Logging level of the logger.')
    parser.add_argument('--log-file', default=None, type=str,
                        help=('Output log to the specific log file, ' +
                              'the log is still printed on stderr.'))
    parser.add_argument('--host-ip', default=None, type=str,
                        help=('Host IP addressed, this is only needed ' +
                              'if the host IP cannot be automatically guessed.'))
    parser.add_argument('--hdfs-tempdir', default='/tmp', type=str,
                        help=('Temporary directory in HDFS, ' +
                              ' only needed in YARN mode.'))
    parser.add_argument('--host-file', default=None, type=str,
                        help=('The file contains the list of hostnames, needed for MPI and ssh.'))
    parser.add_argument('--sge-log-dir', default=None, type=str,
                        help=('Log directory of SGD jobs, only needed in SGE mode.'))
    parser.add_argument(
        '--auto-file-cache', default=True, type=bool,
        help=('Automatically cache files appeared in the command line' +
              'to local executor folder.' +
              ' This will also cause rewritten of all the file names in the command,' +
              ' e.g. `../../kmeans ../kmeans.conf` will be rewritten to `./kmeans kmeans.conf`'))
    parser.add_argument('--files', default=[], action='append',
                        help=('The cached file list which will be copied to local environment,' +
                              ' You may need this option to cache additional files.' +
                              ' You  --auto-file-cache is off'))
    parser.add_argument('--archives', default=[], action='append',
                        help=('Same as cached files,' +
                              ' but corresponds to archieve files that will be unziped locally,' +
                              ' You can use this option to ship python libraries.' +
                              ' Only valid in yarn jobs.'))
    parser.add_argument('--env', action='append', default=[],
                        help='Client and ApplicationMaster environment variables.')
    parser.add_argument('--yarn-app-classpath', type=str,
                        help=('Explicit YARN ApplicationMaster classpath.' +
                              'Can be used to override defaults.'))
    parser.add_argument('--yarn-app-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), os.pardir, 'yarn'),
                        help=('Directory to YARN appmaster. Only used in YARN mode.'))
    parser.add_argument('--ship-libcxx', default=None, type=str,
                        help=('The path to the customized gcc lib folder.' +
                              'You can use this option to ship customized libstdc++' +
                              ' library to the workers.'))
    parser.add_argument('--sync-dst-dir', type=str,
                        help = 'if specificed, it will sync the current \
                        directory into remote machines\'s SYNC_DST_DIR')
    parser.add_argument('command', nargs='+',
                        help='Command to be launched')
    (args, unknown) = parser.parse_known_args(args)
    args.command += unknown

    if args.cluster is None:
        args.cluster = os.getenv('DMLC_SUBMIT_CLUSTER', None)

    if args.cluster is None:
        raise RuntimeError('--cluster is not specified, ' +
                           'you can also specify the default behavior via ' +
                           'environment variable DMLC_SUBMIT_CLUSTER')

    args.worker_memory_mb = get_memory_mb(args.worker_memory)
    args.server_memory_mb = get_memory_mb(args.server_memory)
    return args
