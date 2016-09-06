DMLC Tracker
============
Job submission and tracking script for DMLC. To submit your job to cluster.
Use the following command

```bash
dmlc-submit --mode <cluster-mode> [arguments] [command]
```

DMLC job will start executors, each act as role of worker or server.
It works for both parameter server based jobs as well as rabit allreduce jobs.

Parameters
----------
The following is a list of frequently used arguments available in the dmlc-submit command.
To get full list of arguments, you can run
```bash
dmlc-submit -h
```

- ```--cluster``` string, {'mpi', 'yarn', 'local',  'sge'}, default to ${DMLC_SUBMIT_CLUSTER}
  - Job submission mode.
- ```--num-workers``` integer, required
  - Number of workers in the job.
- ```--num-servers```` integer, default=0
  - Number of servers in the job.
- ```--worker-cores``` integer, default=1
  - Number of cores needed to be allocated for worker job.
- ```--server-cores``` integer, default=1
  -  Number of cores needed to be allocated for server job.
- ```--worker-memory``` string, default='1g'
  - Memory needed for server job.
- ```--server-memory``` string, default='1g'
  - Memory needed for server job.
- ```--jobname``` string, default=auto specify
  - Name of the job.
- ```--queue``` string, default='default'
  - The submission queue we should submit the job to.
- ```--log-level``` string, {INFO, DEBUG}
  - The logging level.
- ```--log-file``` string, default='None'
  - Output log to the specific log file, the log is still printed on stderr.
