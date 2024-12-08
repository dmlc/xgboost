####################################
Automated testing in XGBoost project
####################################

This document collects tips for using the Continuous Integration (CI) service of the XGBoost
project.

**Contents**

.. contents::
  :backlinks: none
  :local:

**************
GitHub Actions
**************
We make the extensive use of `GitHub Actions <https://github.com/features/actions>`_ to host our
CI pipelines. Most of the tests listed in the configuration files run automatically for every
incoming pull requests and every update to branches. A few tests however require manual activation:

* R tests with ``noLD`` option: Run R tests using a custom-built R with compilation flag
  ``--disable-long-double``. See `this page <https://blog.r-hub.io/2019/05/21/nold/>`_ for more
  details about noLD. This is a requirement for keeping XGBoost on CRAN (the R package index).
  To invoke this test suite for a particular pull request, simply add a review comment
  ``/gha run r-nold-test``. (Ordinary comment won't work. It needs to be a review comment.)

*******************************
Self-Hosted Runners with RunsOn
*******************************

`RunsOn <https://runs-on.com/>`_ is a SaaS (Software as a Service) app that lets us to easily create
self-hosted runners to use with GitHub Actions pipelines. RunsOn uses
`Amazon Web Services (AWS) <https://aws.amazon.com/>`_ under the hood to provision runners with
access to various amount of CPUs, memory, and NVIDIA GPUs. Thanks to this app, we are able to test
GPU-accelerated and distributed algorithms of XGBoost while using the familar interface of
GitHub Actions.

In GitHub Actions, jobs run on Microsoft-hosted runners by default.
To opt into self-hosted runners (enabled by RunsOn), we use the following special syntax:

.. code-block:: yaml

  runs-on:
    - runs-on
    - runner=runner-name
    - run-id=${{ github.run_id }}
    - tag=[unique tag that uniquely identifies the job in the GH Action workflow]

where the runner is defined in ``.github/runs-on.yml``.

*********************************************************
Reproduce CI testing environments using Docker containers
*********************************************************
In our CI pipelines, we use Docker containers extensively to package many software packages together.
You can reproduce the same testing environment as the CI pipelines by running Docker locally.

=============
Prerequisites
=============
1. Install Docker: https://docs.docker.com/engine/install/ubuntu/
2. Install NVIDIA Docker runtime:
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.
   The runtime lets you access NVIDIA GPUs inside a Docker container.

.. _build_run_docker_locally:

==============================================
Building and Running Docker containers locally
==============================================
For your convenience, we provide three wrapper scripts:

* ``ops/docker_build.py``: Build a Docker container
* ``ops/docker_build.sh``: Wrapper for ``ops/docker_build.py`` with a more concise interface
* ``ops/docker_run.py``: Run a command inside a Docker container

**To build a Docker container**, invoke ``docker_build.sh`` as follows:

.. code-block:: bash

  export BRANCH_NAME="master"  # Relevant for CI, for local testing, use "master"
  bash ops/docker_build.sh CONTAINER_ID

where ``CONTAINER_ID`` identifies for the container. The wrapper script will look up the YAML file
``ops/docker/ci_container.yml``. For example, when ``CONTAINER_ID`` is set to ``xgb-ci.gpu``,
the script will use the corresponding entry from ``ci_container.yml``:

.. code-block:: yaml

  xgb-ci.gpu:
    container_def: gpu
    build_args:
      CUDA_VERSION_ARG: "12.4.1"
      NCCL_VERSION_ARG: "2.23.4-1"
      RAPIDS_VERSION_ARG: "24.10"

The ``container_def`` entry indicates where the Dockerfile is located. The container
definition will be fetched from ``ops/docker/dockerfile/Dockerfile.CONTAINER_DEF`` where
``CONTAINER_DEF`` is the value of ``container_def`` entry. In this example, the Dockerfile
is ``ops/docker/dockerfile/Dockerfile.gpu``.

The ``build_args`` entry lists all the build arguments for the Docker build. In this example,
the build arguments are:

.. code-block::

  --build-arg CUDA_VERSION_ARG=12.4.1 --build-arg NCCL_VERSION_ARG=2.23.4-1 \
    --build-arg RAPIDS_VERSION_ARG=24.10

The build arguments provide inputs to the ``ARG`` instructions in the Dockerfile.

.. note:: Inspect the logs from the CI pipeline to find what's going on under the hood

  When invoked, ``ops/docker_build.sh`` logs the precise commands that it runs under the hood.
  Using the example above:

  .. code-block:: bash

    # docker_build.sh calls docker_build.py...
    python3 ops/docker_build.py --container-def gpu --container-id xgb-ci.gpu \
      --build-arg CUDA_VERSION_ARG=12.4.1 --build-arg NCCL_VERSION_ARG=2.23.4-1 \
      --build-arg RAPIDS_VERSION_ARG=24.10

    ...

    # .. and docker_build.py in turn calls "docker build"...
    docker build --build-arg CUDA_VERSION_ARG=12.4.1 \
      --build-arg NCCL_VERSION_ARG=2.23.4-1 \
      --build-arg RAPIDS_VERSION_ARG=24.10 \
      --load --progress=plain \
      --ulimit nofile=1024000:1024000 \
      -t xgb-ci.gpu \
      -f ops/docker/dockerfile/Dockerfile.gpu \
      ops/
  
  The logs come in handy when debugging the container builds. In addition, you can change
  the build arguments to make changes to the container.

**To run commands within a Docker container**, invoke ``docker_run.py`` as follows:

.. code-block:: bash

  python3 ops/docker_run.py --container-id "ID of the container" [--use-gpus] \
    -- "command to run inside the container"

where ``--use-gpus`` should be specified to expose NVIDIA GPUs to the Docker container.

For example:

.. code-block:: bash

  # Run without GPU
  python3 ops/docker_run.py --container-id xgb-ci.cpu \
    -- bash ops/script/build_via_cmake.sh

  # Run with NVIDIA GPU
  python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
    -- bash ops/pipeline/test-python-wheel-impl.sh gpu

The ``docker_run.py`` script will convert these commands to the following invocations
of ``docker run``:

.. code-block:: bash

  docker run --rm --pid=host \
    -w /workspace -v /path/to/xgboost:/workspace \
    -e CI_BUILD_UID=<uid> -e CI_BUILD_USER=<user_name> \
    -e CI_BUILD_GID=<gid> -e CI_BUILD_GROUP=<group_name> \
    xgb-ci.cpu \
    bash ops/script/build_via_cmake.sh

  docker run --rm --pid=host --gpus all \
    -w /workspace -v /path/to/xgboost:/workspace \
    -e CI_BUILD_UID=<uid> -e CI_BUILD_USER=<user_name> \
    -e CI_BUILD_GID=<gid> -e CI_BUILD_GROUP=<group_name> \
    xgb-ci.gpu \
    bash ops/pipeline/test-python-wheel-impl.sh gpu

Optionally, you can specify ``--run-args`` to pass extra arguments to ``docker run``:

.. code-block:: bash

  # Allocate extra space in /dev/shm to enable NCCL
  # Also run the container with elevated privileges
  python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
    --run-args='--shm-size=4g --privileged' \
    -- bash ops/pipeline/test-python-wheel-impl.sh gpu

which translates to

.. code-block:: bash

  docker run --rm --pid=host --gpus all \
    -w /workspace -v /path/to/xgboost:/workspace \
    -e CI_BUILD_UID=<uid> -e CI_BUILD_USER=<user_name> \
    -e CI_BUILD_GID=<gid> -e CI_BUILD_GROUP=<group_name> \
    --shm-size=4g --privileged \
    xgb-ci.gpu \
    bash ops/pipeline/test-python-wheel-impl.sh gpu

*******************************************************************
The Lay of the Land: how CI pipelines are organized in the codebase
*******************************************************************
The XGBoost project stores the configuration for its CI pipelines as part of the codebase.
The git repository therefore stores not only the change history for its source code but also
the change history for the CI pipelines.

=================
File Organization
=================

The CI pipelines are organized into the following directories and files:

* ``.github/workflows/``: Definition of CI pipelines, using the GitHub Actions syntax
* ``.github/runs-on.yml``: Configuration for the RunsOn service. Specifies the spec for
  the self-hosted CI runners.
* ``ops/conda_env/``: Definitions for Conda environments
* ``ops/packer/``: Packer scripts to build VM images for Amazon EC2
* ``ops/patch/``: Patch files
* ``ops/pipeline/``: Shell scripts defining CI/CD pipelines. Most of these scripts can be run
  locally (to assist with development and debugging); a few must run in the CI.
* ``ops/script/``: Various utility scripts useful for testing
* ``ops/docker/dockerfile/``: Dockerfiles to define containers
* ``ops/docker/ci_container.yml``: Defines the mapping between Dockerfiles and containers.
  Also specifies the build arguments to be used with each container. See
  :ref:`build_run_docker_locally` to learn how this YAML file is used in the context of
  a container build.
* ``ops/docker_build.*``: Wrapper scripts to build and test CI containers. See
  :ref:`build_run_docker_locally` for the detailed description.

To inspect a given CI pipeline, inspect files in the following order:

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph ci_graph {
      graph [fontname = "monospace"];
      node [fontname = "monospace"];
      edge [fontname = "monospace"];
      0 [label=<.github/workflows/*.yml>, shape=box];
      1 [label=<ops/pipeline/*.sh>, shape=box];
      2 [label=<ops/pipeline/*-impl.sh>, shape=box];
      3 [label=<ops/script/*.sh>, shape=box];
      0 -> 1 [xlabel="Calls"];
      1 -> 2 [xlabel="Calls,\nvia docker_run.py"];
      2 -> 3 [xlabel="Calls"];
      1 -> 3 [xlabel="Calls"];
    }
  """
  Source(source, format='png').render('../_static/ci_graph', view=False)
  Source(source, format='svg').render('../_static/ci_graph', view=False)

.. figure:: ../_static/ci_graph.svg
   :align: center
   :figwidth: 80 %

===================================
Primitives used in the CI pipelines
===================================

------------------------
Build and run containers
------------------------

See :ref:`build_run_docker_locally` to learn about the utility scripts for building and
using containers.

**What's the relationship between the VM image (for Amazon EC2) and the container image?**
In ``ops/packer/`` directory, we define Packer scripts to build VM images for Amazon EC2.
The VM image contains the minimal set of drivers and system software that are needed to
run the containers.

We update container images much more often than VM images. Whereas VM images are
updated sparingly (once in a few months), container images are updated each time a branch
or a pull request is updated. This way, developers can make changes to containers and
see the results of the changes immediately in the CI run.

------------------------------------------
Stash artifacts, to move them between jobs
------------------------------------------

This primitive is useful when one pipeline job needs to consume the output
from another job.
We use `Amazon S3 <https://aws.amazon.com/s3/>`_ to store the stashed files.

**To stash a file**:

.. code-block:: bash

  REMOTE_PREFIX="remote directory to place the artifact(s)"
  bash ops/pipeline/stash-artifacts.sh stash "${REMOTE_PREFIX}" path/to/file

The ``REMOTE_PREFIX`` argument, which is the second command-line argument
for ``stash-artifacts.sh``, specifies the remote directory in which the artifact(s)
should be placed. More precisely, the artifact(s) will be placed in
``s3://{RUNS_ON_S3_BUCKET_CACHE}/cache/{GITHUB_REPOSITORY}/stash/{GITHUB_RUN_ID}/{REMOTE_PREFIX}/``
where ``RUNS_ON_S3_BUCKET_CACHE``, ``GITHUB_REPOSITORY``, and ``GITHUB_RUN_ID`` are set by
the CI. (RunsOn provisions an S3 bucket to stage cache, and its name is stored in the environment
variable ``RUNS_ON_S3_BUCKET_CACHE``.)

You can upload multiple files, possibly with wildcard globbing:

.. code-block:: bash

  REMOTE_PREFIX="build-cuda"
  bash ops/pipeline/stash-artifacts.sh stash "${REMOTE_PREFIX}" \
    build/testxgboost python-package/dist/*.whl

**To unstash a file**:

.. code-block:: bash

  REMOTE_PREFIX="remote directory to place the artifact(s)"
  bash ops/pipeline/stash-artifacts.sh unstash "${REMOTE_PREFIX}" path/to/file

You can also use the wildcard globbing. The script will download the matching artifacts
from the remote directory.

.. code-block:: bash

  REMOTE_PREFIX="build-cuda"
  # Download all files whose path matches the wildcard pattern python-package/dist/*.whl
  bash ops/pipeline/stash-artifacts.sh unstash "${REMOTE_PREFIX}" \
    python-package/dist/*.whl

-----------------------------------------
Custom actions in ``dmlc/xgboost-devops``
-----------------------------------------

XGBoost implements a few custom
`composite actions <https://docs.github.com/en/actions/sharing-automations/creating-actions/creating-a-composite-action>`_
to reduce duplicated code within workflow YAML files. The custom actions are hosted in a separate repository,
`dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_, to make it easy to test changes to the custom actions in
a pull request or a fork.

In a workflow file, we'd refer to ``dmlc/xgboost-devops/{custom-action}@main``. For example:

.. code-block:: yaml

  - uses: dmlc/xgboost-devops/miniforge-setup@main
    with:
      environment-name: cpp_test
      environment-file: ops/conda_env/cpp_test.yml

Each custom action consists of two components:

* Main script (``dmlc/xgboost-devops/{custom-action}/action.yml``): dispatches to a specific version
  of the implementation script (see the next item). The main script clones ``xgboost-devops`` from
  a specified fork at a particular ref, allowing us to easily test changes to the custom action.
* Implementation script (``dmlc/xgboost-devops/impls/{custom-action}/action.yml``): Implements the
  custom script.

This design was inspired by Mike Sarahan's work in
`rapidsai/shared-actions <https://github.com/rapidsai/shared-actions>`_.
