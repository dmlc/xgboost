####################################
Automated testing in XGBoost project
####################################

This document collects tips for using the Continuous Integration (CI) service of the XGBoost
project.

**Contents**

.. contents::
  :backlinks: none
  :local:

****************
Tips for testing
****************

====================================
Running R tests with ``noLD`` option
====================================
You can run R tests using a custom-built R with compilation flag
``--disable-long-double``. See `this page <https://blog.r-hub.io/2019/05/21/nold/>`_ for more
details about noLD. This is a requirement for keeping XGBoost on CRAN (the R package index).
Unlike other tests, this test must be invoked manually. Simply add a review comment
``/gha run r-nold-test`` to a pull request to kick off the test.
(Ordinary comment won't work. It needs to be a review comment.)

===============================
Making changes to CI containers
===============================
Many of the CI pipelines use Docker containers to ensure consistent testing environment
with a variety of software packages. We have a separate repo,
`dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_, to host the logic for
building and publishing CI containers.

To make changes to the CI container, carry out the following steps:

1. Identify which container needs updating. Example:
   ``492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main``
2. Clone `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_ and make changes to the
   corresponding Dockerfile. Example: ``containers/dockerfile/Dockerfile.gpu``.
3. Locally build the container, to ensure that the container successfully builds.
   Consult :ref:`build_run_docker_locally` for this step.
4. Submit a pull request to `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_ with
   the proposed changes to the Dockerfile. Make note of the pull request number. Example: ``#204``
5. Clone `dmlc/xgboost <https://github.com/dmlc/xgboost>`_. Locate the file
   ``ops/pipeline/get-image-tag.sh``, which should have a single line

   .. code-block:: bash

     IMAGE_TAG=main

   To use the new container, revise the file as follows:

   .. code-block:: bash

     IMAGE_TAG=PR-XX

   where ``XX`` is the pull request number.

6. Now submit a pull request to `dmlc/xgboost <https://github.com/dmlc/xgboost>`_. The CI will
   run tests using the new container. Verify that all tests pass.
7. Merge the pull request in ``dmlc/xgboost-devops``. Wait until the CI completes on the ``main`` branch.
8. Go back to the the pull request for ``dmlc/xgboost`` and change ``ops/pipeline/get-image-tag.sh``
   back to ``IMAGE_TAG=main``.
9. Merge the pull request in ``dmlc/xgboost``.

.. _build_run_docker_locally:

===========================================
Reproducing CI testing environments locally
===========================================
You can reproduce the same testing environment as the CI pipelines by building and running Docker
containers locally.

**Prerequisites**

1. Install Docker: https://docs.docker.com/engine/install/ubuntu/
2. Install NVIDIA Docker runtime:
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.
   The runtime lets you access NVIDIA GPUs inside a Docker container.

---------------------------
To build a Docker container
---------------------------
Clone the repository `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_
and invoke ``containers/docker_build.sh`` as follows:

.. code-block:: bash

  # The following env vars are only relevant for CI
  # For local testing, set them to "main"
  export GITHUB_SHA="main"
  export BRANCH_NAME="main"
  bash containers/docker_build.sh IMAGE_REPO

where ``IMAGE_REPO`` is the name of the container image. The wrapper script will look up the
YAML file ``containers/ci_container.yml``. For example, when ``IMAGE_REPO`` is set to
``xgb-ci.gpu``, the script will use the corresponding entry from
``containers/ci_container.yml``:

.. code-block:: yaml

  xgb-ci.gpu:
    container_def: gpu
    build_args:
      CUDA_VERSION_ARG: "12.4.1"
      NCCL_VERSION_ARG: "2.23.4-1"
      RAPIDS_VERSION_ARG: "24.10"

The ``container_def`` entry indicates where the Dockerfile is located. The container
definition will be fetched from ``containers/dockerfile/Dockerfile.CONTAINER_DEF`` where
``CONTAINER_DEF`` is the value of ``container_def`` entry. In this example, the Dockerfile
is ``containers/dockerfile/Dockerfile.gpu``.

The ``build_args`` entry lists all the build arguments for the Docker build. In this example,
the build arguments are:

.. code-block::

  --build-arg CUDA_VERSION_ARG=12.4.1 --build-arg NCCL_VERSION_ARG=2.23.4-1 \
    --build-arg RAPIDS_VERSION_ARG=24.10

The build arguments provide inputs to the ``ARG`` instructions in the Dockerfile.

When ``containers/docker_build.sh`` completes, you will have access to the container with the
(fully qualified) URI ``492475357299.dkr.ecr.us-west-2.amazonaws.com/[image_repo]:main``.
The prefix ``492475357299.dkr.ecr.us-west-2.amazonaws.com/`` was added so that
the container could later be uploaded to AWS Elastic Container Registry (ECR),
a private Docker registry.

-----------------------------------------
To run commands within a Docker container
-----------------------------------------
Invoke ``ops/docker_run.py`` from the main ``dmlc/xgboost`` repo as follows:

.. code-block:: bash

  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/[image_repo]:[image_tag] \
    [--use-gpus] \
    -- "command to run inside the container"

where ``--use-gpus`` should be specified to expose NVIDIA GPUs to the Docker container.

For example:

.. code-block:: bash

  # Run without GPU
  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.cpu:main \
    -- bash ops/pipeline/build-cpu-impl.sh cpu

  # Run with NVIDIA GPU
  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    --use-gpus \
    -- bash ops/pipeline/test-python-wheel-impl.sh gpu

Optionally, you can specify ``--run-args`` to pass extra arguments to ``docker run``:

.. code-block:: bash

  # Allocate extra space in /dev/shm to enable NCCL
  # Also run the container with elevated privileges
  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    --use-gpus \
    --run-args='--shm-size=4g --privileged' \
    -- bash ops/pipeline/test-python-wheel-impl.sh gpu

See :ref:`ci_container_infra` to read about how containers are built and managed in the CI pipelines.

--------------------------------------------
Examples: useful tasks for local development
--------------------------------------------

* Build XGBoost with GPU support + package it as a Python wheel

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.gpu_build_rockylinux8:main \
      -- ops/pipeline/build-cuda-impl.sh

* Run Python tests

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.cpu:main \
      -- ops/pipeline/test-python-wheel-impl.sh cpu

* Run Python tests with GPU algorithm

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.gpu:main \
      --use-gpus \
      -- ops/pipeline/test-python-wheel-impl.sh gpu

* Run Python tests with GPU algorithm, with multiple GPUs

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.gpu:main \
      --use-gpus \
      --run-args='--shm-size=4g' \
      -- ops/pipeline/test-python-wheel-impl.sh mgpu
      # --shm-size=4g is needed for multi-GPU algorithms to function

* Build and test JVM packages

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    export SCALA_VERSION=2.12  # Specify Scala version (2.12 or 2.13)
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.jvm:main \
      --run-args "-e SCALA_VERSION" \
      -- ops/pipeline/build-test-jvm-packages-impl.sh

* Build and test JVM packages, with GPU support

  .. code-block:: bash

    export DOCKER_REGISTRY=492475357299.dkr.ecr.us-west-2.amazonaws.com
    export SCALA_VERSION=2.12  # Specify Scala version (2.12 or 2.13)
    export USE_CUDA=1
    python3 ops/docker_run.py \
      --image-uri ${DOCKER_REGISTRY}/xgb-ci.jvm_gpu_build:main \
      --use-gpus \
      --run-args "-e SCALA_VERSION -e USE_CUDA --shm-size=4g" \
      -- ops/pipeline/build-test-jvm-packages-impl.sh
      # --shm-size=4g is needed for multi-GPU algorithms to function

*****************************
Tour of the CI infrastructure
*****************************

==============
GitHub Actions
==============
We make the extensive use of `GitHub Actions <https://github.com/features/actions>`_ to host our
CI pipelines. Most of the tests listed in the configuration files run automatically for every
incoming pull requests and every update to branches.

===============================
Self-Hosted Runners with RunsOn
===============================
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

===================================================================
The Lay of the Land: how CI pipelines are organized in the codebase
===================================================================
The XGBoost project stores the configuration for its CI pipelines as part of the codebase.
The git repository therefore stores not only the change history for its source code but also
the change history for the CI pipelines.

The CI pipelines are organized into the following directories and files:

* ``.github/workflows/``: Definition of CI pipelines, using the GitHub Actions syntax
* ``.github/runs-on.yml``: Configuration for the RunsOn service. Specifies the spec for
  the self-hosted CI runners.
* ``ops/conda_env/``: Definitions for Conda environments
* ``ops/patch/``: Patch files
* ``ops/pipeline/``: Shell scripts defining CI/CD pipelines. Most of these scripts can be run
  locally (to assist with development and debugging); a few must run in the CI.
* ``ops/script/``: Various utility scripts useful for testing
* ``ops/docker_run.py``: Wrapper script to run commands inside a container

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

Many of the CI pipelines use Docker containers to ensure consistent testing environment
with a variety of software packages. We have a separate repo,
`dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_, that
hosts the code for building the CI containers. The repository is organized as follows:

* ``actions/``: Custom actions to be used with GitHub Actions. See :ref:`custom_actions`
  for more details.
* ``containers/dockerfile/``: Dockerfiles to define containers
* ``containers/ci_container.yml``: Defines the mapping between Dockerfiles and containers.
  Also specifies the build arguments to be used with each container.
* ``containers/docker_build.{py,sh}``: Wrapper scripts to build and test CI containers.
* ``vm_images/``: Defines bootstrap scripts to build VM images for Amazon EC2. See
  :ref:`vm_images` to learn about how VM images relate to container images.

See :ref:`build_run_docker_locally` to learn about the utility scripts for building and
using containers.

===========================================
Artifact sharing between jobs via Amazon S3
===========================================

We make artifacts from one workflow job available to another job, by uploading the
artifacts to `Amazon S3 <https://aws.amazon.com/s3/>`_. In the CI, we utilize the
script ``ops/pipeline/manage-artifacts.py`` to coordinate artifact sharing.

**To upload files to S3**: In the workflow YAML, add the following lines:

.. code-block:: yaml

  - name: Upload files to S3
    run: |
      REMOTE_PREFIX="remote directory to place the artifact(s)"
      python3 ops/pipeline/manage-artifacts.py upload \
        --s3-bucket ${{ env.RUNS_ON_S3_BUCKET_CACHE }} \
        --prefix cache/${{ github.run_id }}/${REMOTE_PREFIX} \
        path/to/file

The ``--prefix`` argument specifies the remote directory in which the artifact(s)
should be placed. The artifact(s) will be placed in
``s3://{RUNS_ON_S3_BUCKET_CACHE}/cache/{GITHUB_RUN_ID}/{REMOTE_PREFIX}/``
where ``RUNS_ON_S3_BUCKET_CACHE`` and ``GITHUB_RUN_ID`` are set by the CI.

You can upload multiple files, possibly with wildcard globbing:

.. code-block:: yaml

  - name: Upload files to S3
    run: |
      python3 ops/pipeline/manage-artifacts.py upload \
        --s3-bucket ${{ env.RUNS_ON_S3_BUCKET_CACHE }} \
        --prefix cache/${{ github.run_id }}/build-cuda \
        build/testxgboost python-package/dist/*.whl

**To download files from S3**: In the workflow YAML, add the following lines:

.. code-block:: yaml

  - name: Download files from S3
    run: |
      REMOTE_PREFIX="remote directory where the artifact(s) were placed"
      python3 ops/pipeline/manage-artifacts.py download \
        --s3-bucket ${{ env.RUNS_ON_S3_BUCKET_CACHE }} \
        --prefix cache/${{ github.run_id }}/${REMOTE_PREFIX} \
        --dest-dir path/to/destination_directory \
        artifacts

You can also use the wildcard globbing. The script will locate all artifacts
under the given prefix that matches the wildcard pattern.

.. code-block:: yaml

  - name: Download files from S3
    run: |
      # Locate all artifacts with name *.whl under prefix
      # cache/${GITHUB_RUN_ID}/${REMOTE_PREFIX} and
      # download them to wheelhouse/.
      python3 ops/pipeline/manage-artifacts.py download \
        --s3-bucket ${{ env.RUNS_ON_S3_BUCKET_CACHE }} \
        --prefix cache/${{ github.run_id }}/${REMOTE_PREFIX} \
        --dest-dir wheelhouse/ \
        *.whl

.. _custom_actions:

=================================
Custom actions for GitHub Actions
=================================

XGBoost implements a few custom
`composite actions <https://docs.github.com/en/actions/sharing-automations/creating-actions/creating-a-composite-action>`_
to reduce duplicated code within workflow YAML files. The custom actions are hosted in a separate repository,
`dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_, to make it easy to test changes to the custom actions in
a pull request or a fork.

In a workflow file, we'd refer to ``dmlc/xgboost-devops/actions/{custom-action}@main``. For example:

.. code-block:: yaml

  - uses: dmlc/xgboost-devops/actions/miniforge-setup@main
    with:
      environment-name: cpp_test
      environment-file: ops/conda_env/cpp_test.yml

Each custom action consists of two components:

* Main script (``dmlc/xgboost-devops/actions/{custom-action}/action.yml``): dispatches to a specific version
  of the implementation script (see the next item). The main script clones ``xgboost-devops`` from
  a specified fork at a particular ref, allowing us to easily test changes to the custom action.
* Implementation script (``dmlc/xgboost-devops/actions/impls/{custom-action}/action.yml``): Implements the
  custom script.

This design was inspired by Mike Sarahan's work in
`rapidsai/shared-actions <https://github.com/rapidsai/shared-actions>`_.


.. _ci_container_infra:

=============================================================
Infra for building and publishing CI containers and VM images
=============================================================

--------------------------
Notes on Docker containers
--------------------------
**CI pipeline for containers**

The `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_ repo hosts a CI pipeline to build new
Docker containers at a regular schedule. New containers are built in the following occasions:

* New commits are added to the ``main`` branch of ``dmlc/xgboost-devops``.
* New pull requests are submitted to ``dmlc/xgboost-devops``.
* Every week, at a set day and hour.

This setup ensures that the CI containers remain up-to-date.

**How wrapper scripts work**

The wrapper scripts ``docker_build.sh``, ``docker_build.py`` (in ``dmlc/xgboost-devops``) and ``docker_run.py``
(in ``dmlc/xgboost``) are designed to transparently log what commands are being carried out under the hood.
For example, when you run ``bash containers/docker_build.sh xgb-ci.gpu``, the logs will show the following:

.. code-block:: bash

  # docker_build.sh calls docker_build.py...
  python3 containers/docker_build.py --container-def gpu \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    --build-arg CUDA_VERSION_ARG=12.4.1 --build-arg NCCL_VERSION_ARG=2.23.4-1 \
    --build-arg RAPIDS_VERSION_ARG=24.10

  ...

  # .. and docker_build.py in turn calls "docker build"...
  docker build --build-arg CUDA_VERSION_ARG=12.4.1 \
    --build-arg NCCL_VERSION_ARG=2.23.4-1 \
    --build-arg RAPIDS_VERSION_ARG=24.10 \
    --load --progress=plain \
    --ulimit nofile=1024000:1024000 \
    -t 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    -f containers/dockerfile/Dockerfile.gpu \
    containers/

The logs come in handy when debugging the container builds.

Here is an example with ``docker_run.py``:

.. code-block:: bash

  # Run without GPU
  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.cpu:main \
    -- bash ops/pipeline/build-cpu-impl.sh cpu

  # Run with NVIDIA GPU
  # Allocate extra space in /dev/shm to enable NCCL
  # Also run the container with elevated privileges
  python3 ops/docker_run.py \
    --image-uri 492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    --use-gpus \
    --run-args='--shm-size=4g --privileged' \
    -- bash ops/pipeline/test-python-wheel-impl.sh gpu

which are translated to the following ``docker run`` invocations:

.. code-block:: bash

  docker run --rm --pid=host \
    -w /workspace -v /path/to/xgboost:/workspace \
    -e CI_BUILD_UID=<uid> -e CI_BUILD_USER=<user_name> \
    -e CI_BUILD_GID=<gid> -e CI_BUILD_GROUP=<group_name> \
    492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.cpu:main \
    bash ops/pipeline/build-cpu-impl.sh cpu

  docker run --rm --pid=host --gpus all \
    -w /workspace -v /path/to/xgboost:/workspace \
    -e CI_BUILD_UID=<uid> -e CI_BUILD_USER=<user_name> \
    -e CI_BUILD_GID=<gid> -e CI_BUILD_GROUP=<group_name> \
    --shm-size=4g --privileged \
    492475357299.dkr.ecr.us-west-2.amazonaws.com/xgb-ci.gpu:main \
    bash ops/pipeline/test-python-wheel-impl.sh gpu


.. _vm_images:

------------------
Notes on VM images
------------------

In the ``vm_images/`` directory of `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_,
we define Packer scripts to build images for Virtual Machines (VM) on
`Amazon EC2 <https://aws.amazon.com/ec2/>`_.
The VM image contains the minimal set of drivers and system software that are needed to
run the containers.

We update container images much more often than VM images. Whereas it takes only 10 minutes to
build a new container image, it takes 1-2 hours to build a new VM image.

To enable quick development iteration cycle, we place the most of
the development environment in containers and keep VM images small.
Packages need for testing should be baked into containers, not VM images.
Developers can make changes to containers and see the results of the changes quickly.

.. note:: Special note for the Windows platform

  We do not use containers when testing XGBoost on Windows. All software must be baked into
  the VM image. Containers are not used because
  `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html>`_
  does not yet support Windows natively.

The `dmlc/xgboost-devops <https://github.com/dmlc/xgboost-devops>`_ repo hosts a CI pipeline to build new
VM images at a regular schedule (currently monthly).
