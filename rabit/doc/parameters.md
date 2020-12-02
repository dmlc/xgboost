Parameters
==========
This section list all the parameters that can be passed to rabit::Init function as argv.
All the parameters are passed in as string in format of ``parameter-name=parameter-value``.
In most setting these parameters have default value or will be automatically detected,
and do not need to be manually configured.

* rabit_tracker_uri [passed in automatically by tracker]
  - The uri/ip of rabit tracker
* rabit_tracker_port [passed in automatically by tracker]
  - The port of rabit tracker
* rabit_task_id [automatically detected]
  - The unique identifier of computing process
  - When running on hadoop, this is automatically extracted from enviroment variable
* rabit_reduce_buffer [default = 256MB]
  - The memory buffer used to store intermediate result of reduction
  - Format "digits + unit", can be 128M, 1G
* rabit_global_replica [default = 5]
  - Number of replication copies of result kept for each Allreduce/Broadcast call
* rabit_local_replica [default = 2]
  - Number of replication of local model in check point
