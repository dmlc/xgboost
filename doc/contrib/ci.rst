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
The configuration files are located under the directory
`.github/workflows <https://github.com/dmlc/xgboost/tree/master/.github/workflows>`_.

Most of the tests listed in the configuration files run automatically for every incoming pull
requests and every update to branches. A few tests however require manual activation:

* R tests with ``noLD`` option: Run R tests using a custom-built R with compilation flag
  ``--disable-long-double``. See `this page <https://blog.r-hub.io/2019/05/21/nold/>`_ for more
  details about noLD. This is a requirement for keeping XGBoost on CRAN (the R package index).
  To invoke this test suite for a particular pull request, simply add a review comment
  ``/gha run r-nold-test``. (Ordinary comment won't work. It needs to be a review comment.)

GitHub Actions is also used to build Python wheels targeting MacOS Intel and Apple Silicon. See
`.github/workflows/python_wheels.yml
<https://github.com/dmlc/xgboost/tree/master/.github/workflows/python_wheels.yml>`_. The
``python_wheels`` pipeline sets up environment variables prefixed ``CIBW_*`` to indicate the target
OS and processor. The pipeline then invokes the script ``build_python_wheels.sh``, which in turns
calls ``cibuildwheel`` to build the wheel. The ``cibuildwheel`` is a library that sets up a
suitable Python environment for each OS and processor target. Since we don't have Apple Silion
machine in GitHub Actions, cross-compilation is needed; ``cibuildwheel`` takes care of the complex
task of cross-compiling a Python wheel. (Note that ``cibuildwheel`` will call
``setup.py bdist_wheel``. Since XGBoost has a native library component, ``setup.py`` contains
a glue code to call CMake and a C++ compiler to build the native library on the fly.)

*******************************
Elastic CI Stack with BuildKite
*******************************

`BuildKite <https://buildkite.com/home>`_ is a SaaS (Software as a Service) platform that orchestrates
cloud machines to host CI pipelines. The BuildKite platform allows us to define cloud resources in
a declarative fashion. Every configuration step is now documented explicitly as code.

**Prerequisite**: You should have some knowledge of `CloudFormation <https://aws.amazon.com/cloudformation/>`_.
CloudFormation lets us define a stack of cloud resources (EC2 machines, Lambda functions, S3 etc) using
a single YAML file.

**Prerequisite**: Gain access to the XGBoost project's AWS account (``admin@xgboost-ci.net``), and then
set up a credential pair in order to provision resources on AWS. See
`Creating an IAM user in your AWS account <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html>`_.

* Option 1. Give full admin privileges to your IAM user. This is the simplest option.
* Option 2. Give limited set of permissions to your IAM user, to reduce the possibility of messing up other resources.
  For this, use the script ``tests/buildkite/infrastructure/service-user/create_service_user.py``.

=====================
Worker Image Pipeline
=====================
Building images for worker machines used to be a chore: you'd provision an EC2 machine, SSH into it, and
manually install the necessary packages. This process is not only laborous but also error-prone. You may
forget to install a package or change a system configuration.

No more. Now we have an automated pipeline for building images for worker machines.

* Run ``tests/buildkite/infrastructure/worker-image-pipeline/create_worker_image_pipelines.py`` in order to provision
  CloudFormation stacks named ``buildkite-linux-amd64-gpu-worker`` and ``buildkite-windows-gpu-worker``. They are
  pipelines that create AMIs (Amazon Machine Images) for Linux and Windows workers, respectively.
* Navigate to the CloudFormation web console to verify that the image builder pipelines have been provisioned. It may
  take some time.
* Once they pipelines have been fully provisioned, run the script
  ``tests/buildkite/infrastructure/worker-image-pipeline/run_pipelines.py`` to execute the pipelines. New AMIs will be
  uploaded to the EC2 service. You can locate them in the EC2 console.
* Make sure to modify ``tests/buildkite/infrastructure/aws-stack-creator/metadata.py`` to use the correct AMI IDs.
  (For ``linux-amd64-cpu`` and ``linux-arm64-cpu``, use the AMIs provided by BuildKite. Consult the ``AWSRegion2AMI``
  section of https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml.)

======================
EC2 Autoscaling Groups
======================
In EC2, you can create auto-scaling groups, where you can dynamically adjust the number of worker instances according to
workload. When a pull request is submitted, the following steps take place:

1. GitHub sends a signal to the registered webhook, which connects to the BuildKite server.
2. BuildKite sends a signal to a `Lambda <https://aws.amazon.com/lambda/>`_ function named ``Autoscaling``.
3. The Lambda function sends a signal to the auto-scaling group. The group scales up and adds additional worker instances.
4. New worker instances run the test jobs. Test results are reported back to BuildKite.
5. When the test jobs complete, BuildKite sends a signal to ``Autoscaling``, which in turn requests the autoscaling group
   to scale down. Idle worker instances are shut down.

To set up the auto-scaling group, run the script ``tests/buildkite/infrastructure/aws-stack-creator/create_stack.py``.
Check the CloudFormation web console to verify successful provision of auto-scaling groups.