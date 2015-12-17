Trackers
=====
This folder contains tracker scripts that can be used to submit yarn jobs to different platforms,
the example guidelines are in the script themselfs

***Supported Platforms***
* Local demo: [rabit_demo.py](rabit_demo.py)
* MPI: [rabit_mpi.py](rabit_mpi.py)
* Yarn (Hadoop): [rabit_yarn.py](rabit_yarn.py)
  - It is also possible to submit via hadoop streaming with rabit_hadoop_streaming.py
  - However, it is higly recommended to use rabit_yarn.py because this will allocate resources more precisely and fits machine learning scenarios
* Sun Grid engine: [rabit_sge.py](rabit_sge.py)
