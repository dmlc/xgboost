#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build.sh'

// Which CUDA version to use when building reference distribution wheel
ref_cuda_ver = '10.0'

import groovy.transform.Field

@Field
def commit_id   // necessary to pass a variable from one stage to another

pipeline {
  // Each stage specify its own agent
  agent none

  environment {
    DOCKER_CACHE_ECR_ID = '492475357299'
    DOCKER_CACHE_ECR_REGION = 'us-west-2'
  }

  // Setup common job properties
  options {
    ansiColor('xterm')
    timestamps()
    timeout(time: 240, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '10'))
    preserveStashes()
  }

  // Build stages
  stages {
    stage('Jenkins Linux: Initialize') {
      agent { label 'job_initializer' }
      steps {
        script {
          checkoutSrcs()
          commit_id = "${GIT_COMMIT}"
        }
        sh 'python3 tests/jenkins_get_approval.py'
        stash name: 'srcs'
        milestone ordinal: 1
      }
    }
    stage('Jenkins Linux: Formatting Check') {
      agent none
      steps {
        script {
          parallel ([
            'clang-tidy': { ClangTidy() },
            'lint': { Lint() },
            'sphinx-doc': { SphinxDoc() },
            'doxygen': { Doxygen() }
          ])
        }
        milestone ordinal: 2
      }
    }
    stage('Jenkins Linux: Build') {
      agent none
      steps {
        script {
          parallel ([
            'build-cpu': { BuildCPU() },
            'build-cpu-rabit-mock': { BuildCPUMock() },
            'build-cpu-non-omp': { BuildCPUNonOmp() },
            // Build reference, distribution-ready Python wheel with CUDA 10.0
            // using CentOS 6 image
            'build-gpu-cuda10.0': { BuildCUDA(cuda_version: '10.0') },
            // The build-gpu-* builds below use Ubuntu image
            'build-gpu-cuda10.1': { BuildCUDA(cuda_version: '10.1') },
            'build-gpu-cuda10.2': { BuildCUDA(cuda_version: '10.2', build_rmm: true) },
            'build-gpu-cuda11.0': { BuildCUDA(cuda_version: '11.0') },
            'build-jvm-packages-gpu-cuda10.0': { BuildJVMPackagesWithCUDA(spark_version: '3.0.0', cuda_version: '10.0') },
            'build-jvm-packages': { BuildJVMPackages(spark_version: '3.0.0') },
            'build-jvm-doc': { BuildJVMDoc() }
          ])
        }
        milestone ordinal: 3
      }
    }
    stage('Jenkins Linux: Test') {
      agent none
      steps {
        script {
          parallel ([
            'test-python-cpu': { TestPythonCPU() },
            // artifact_cuda_version doesn't apply to RMM tests; RMM tests will always match CUDA version between artifact and host env
            'test-python-gpu-cuda10.2': { TestPythonGPU(artifact_cuda_version: '10.0', host_cuda_version: '10.2', test_rmm: true) },
            'test-python-gpu-cuda11.0-cross': { TestPythonGPU(artifact_cuda_version: '10.0', host_cuda_version: '11.0') },
            'test-python-gpu-cuda11.0': { TestPythonGPU(artifact_cuda_version: '11.0', host_cuda_version: '11.0') },
            'test-python-mgpu-cuda10.2': { TestPythonGPU(artifact_cuda_version: '10.0', host_cuda_version: '10.2', multi_gpu: true, test_rmm: true) },
            'test-cpp-gpu-cuda10.2': { TestCppGPU(artifact_cuda_version: '10.2', host_cuda_version: '10.2', test_rmm: true) },
            'test-cpp-gpu-cuda11.0': { TestCppGPU(artifact_cuda_version: '11.0', host_cuda_version: '11.0') },
            'test-jvm-jdk8-cuda10.0': { CrossTestJVMwithJDKGPU(artifact_cuda_version: '10.0', host_cuda_version: '10.0') },
            'test-jvm-jdk8': { CrossTestJVMwithJDK(jdk_version: '8', spark_version: '3.0.0') },
            'test-jvm-jdk11': { CrossTestJVMwithJDK(jdk_version: '11') },
            'test-jvm-jdk12': { CrossTestJVMwithJDK(jdk_version: '12') },
            'test-r-3.5.3': { TestR(use_r35: true) }
          ])
        }
        milestone ordinal: 4
      }
    }
    stage('Jenkins Linux: Deploy') {
      agent none
      steps {
        script {
          parallel ([
            'deploy-jvm-packages': { DeployJVMPackages(spark_version: '3.0.0') }
          ])
        }
        milestone ordinal: 5
      }
    }
  }
}

// check out source code from git
def checkoutSrcs() {
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
    }
  }
}

def GetCUDABuildContainerType(cuda_version) {
  return (cuda_version == ref_cuda_ver) ? 'gpu_build_centos6' : 'gpu_build'
}

def ClangTidy() {
  node('linux && cpu_build') {
    unstash name: 'srcs'
    echo "Running clang-tidy job..."
    def container_type = "clang_tidy"
    def docker_binary = "docker"
    def dockerArgs = "--build-arg CUDA_VERSION=10.1"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${dockerArgs} python3 tests/ci_build/tidy.py
    """
    deleteDir()
  }
}

def Lint() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Running lint..."
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} bash -c "source activate cpu_test && make lint"
    """
    deleteDir()
  }
}

def SphinxDoc() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Running sphinx-doc..."
    def container_type = "cpu"
    def docker_binary = "docker"
    def docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='-e SPHINX_GIT_BRANCH=${BRANCH_NAME}'"
    sh """#!/bin/bash
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} bash -c "source activate cpu_test && make -C doc html"
    """
    deleteDir()
  }
}

def Doxygen() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Running doxygen..."
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/doxygen.sh ${BRANCH_NAME}
    """
    if (env.BRANCH_NAME == 'master' || env.BRANCH_NAME.startsWith('release')) {
      echo 'Uploading doc...'
      s3Upload file: "build/${BRANCH_NAME}.tar.bz2", bucket: 'xgboost-docs', acl: 'PublicRead', path: "doxygen/${BRANCH_NAME}.tar.bz2"
    }
    deleteDir()
  }
}

def BuildCPU() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build CPU"
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} rm -fv dmlc-core/include/dmlc/build_config_default.h
      # This step is not necessary, but here we include it, to ensure that DMLC_CORE_USE_CMAKE flag is correctly propagated
      # We want to make sure that we use the configured header build/dmlc/build_config.h instead of include/dmlc/build_config_default.h.
      # See discussion at https://github.com/dmlc/xgboost/issues/5510
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_via_cmake.sh -DPLUGIN_LZ4=ON -DPLUGIN_DENSE_PARSER=ON
    ${dockerRun} ${container_type} ${docker_binary} build/testxgboost
    """
    // Sanitizer test
    def docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='-e ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer -e ASAN_OPTIONS=symbolize=1 -e UBSAN_OPTIONS=print_stacktrace=1:log_path=ubsan_error.log --cap-add SYS_PTRACE'"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_via_cmake.sh -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak;undefined" \
      -DCMAKE_BUILD_TYPE=Debug -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} build/testxgboost
    """

    stash name: 'xgboost_cli', includes: 'xgboost'
    deleteDir()
  }
}

def BuildCPUMock() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build CPU with rabit mock"
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_mock_cmake.sh
    """
    echo 'Stashing rabit C++ test executable (xgboost)...'
    stash name: 'xgboost_rabit_tests', includes: 'xgboost'
    deleteDir()
  }
}

def BuildCPUNonOmp() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build CPU without OpenMP"
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_via_cmake.sh -DUSE_OPENMP=OFF
    """
    echo "Running Non-OpenMP C++ test..."
    sh """
    ${dockerRun} ${container_type} ${docker_binary} build/testxgboost
    """
    deleteDir()
  }
}

def BuildCUDA(args) {
  node('linux && cpu_build') {
    unstash name: 'srcs'
    echo "Build with CUDA ${args.cuda_version}"
    def container_type = GetCUDABuildContainerType(args.cuda_version)
    def docker_binary = "docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
    def arch_flag = ""
    if (env.BRANCH_NAME != 'master' && !(env.BRANCH_NAME.startsWith('release'))) {
      arch_flag = "-DGPU_COMPUTE_VER=75"
    }
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_NCCL=ON -DOPEN_MP:BOOL=ON -DHIDE_CXX_SYMBOLS=ON ${arch_flag}
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} bash -c "cd python-package && rm -rf dist/* && python setup.py bdist_wheel --universal"
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} python tests/ci_build/rename_whl.py python-package/dist/*.whl ${commit_id} manylinux2010_x86_64
    """
    echo 'Stashing Python wheel...'
    stash name: "xgboost_whl_cuda${args.cuda_version}", includes: 'python-package/dist/*.whl'
    if (args.cuda_version == ref_cuda_ver && (env.BRANCH_NAME == 'master' || env.BRANCH_NAME.startsWith('release'))) {
      echo 'Uploading Python wheel...'
      path = ("${BRANCH_NAME}" == 'master') ? '' : "${BRANCH_NAME}/"
      s3Upload bucket: 'xgboost-nightly-builds', path: path, acl: 'PublicRead', workingDir: 'python-package/dist', includePathPattern:'**/*.whl'
    }
    echo 'Stashing C++ test executable (testxgboost)...'
    stash name: "xgboost_cpp_tests_cuda${args.cuda_version}", includes: 'build/testxgboost'
    if (args.build_rmm) {
      echo "Build with CUDA ${args.cuda_version} and RMM"
      container_type = "rmm"
      docker_binary = "docker"
      docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
      sh """
      rm -rf build/
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_via_cmake.sh --conda-env=gpu_test -DUSE_CUDA=ON -DUSE_NCCL=ON -DPLUGIN_RMM=ON ${arch_flag}
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} bash -c "cd python-package && rm -rf dist/* && python setup.py bdist_wheel --universal"
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} python tests/ci_build/rename_whl.py python-package/dist/*.whl ${commit_id} manylinux2010_x86_64
      """
      echo 'Stashing Python wheel...'
      stash name: "xgboost_whl_rmm_cuda${args.cuda_version}", includes: 'python-package/dist/*.whl'
      echo 'Stashing C++ test executable (testxgboost)...'
      stash name: "xgboost_cpp_tests_rmm_cuda${args.cuda_version}", includes: 'build/testxgboost'
    }
    deleteDir()
  }
}

def BuildJVMPackagesWithCUDA(args) {
  node('linux && gpu') {
    unstash name: 'srcs'
    echo "Build XGBoost4J-Spark with Spark ${args.spark_version}, CUDA ${args.cuda_version}"
    def container_type = "jvm_gpu_build"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
    def arch_flag = ""
    if (env.BRANCH_NAME != 'master' && !(env.BRANCH_NAME.startsWith('release'))) {
      arch_flag = "-DGPU_COMPUTE_VER=75"
    }
    // Use only 4 CPU cores
    def docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='--cpuset-cpus 0-3'"
    sh """
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_jvm_packages.sh ${args.spark_version} -Duse.cuda=ON $arch_flag
    """
    echo "Stashing XGBoost4J JAR with CUDA ${args.cuda_version} ..."
    stash name: 'xgboost4j_jar_gpu', includes: "jvm-packages/xgboost4j/target/*.jar,jvm-packages/xgboost4j-spark/target/*.jar,jvm-packages/xgboost4j-example/target/*.jar"
    deleteDir()
  }
}

def BuildJVMPackages(args) {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build XGBoost4J-Spark with Spark ${args.spark_version}"
    def container_type = "jvm"
    def docker_binary = "docker"
    // Use only 4 CPU cores
    def docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='--cpuset-cpus 0-3'"
    sh """
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_jvm_packages.sh ${args.spark_version}
    """
    echo 'Stashing XGBoost4J JAR...'
    stash name: 'xgboost4j_jar', includes: "jvm-packages/xgboost4j/target/*.jar,jvm-packages/xgboost4j-spark/target/*.jar,jvm-packages/xgboost4j-example/target/*.jar"
    deleteDir()
  }
}

def BuildJVMDoc() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Building JVM doc..."
    def container_type = "jvm"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_jvm_doc.sh ${BRANCH_NAME}
    """
    if (env.BRANCH_NAME == 'master' || env.BRANCH_NAME.startsWith('release')) {
      echo 'Uploading doc...'
      s3Upload file: "jvm-packages/${BRANCH_NAME}.tar.bz2", bucket: 'xgboost-docs', acl: 'PublicRead', path: "${BRANCH_NAME}.tar.bz2"
    }
    deleteDir()
  }
}

def TestPythonCPU() {
  node('linux && cpu') {
    unstash name: "xgboost_whl_cuda${ref_cuda_ver}"
    unstash name: 'srcs'
    unstash name: 'xgboost_cli'
    echo "Test Python CPU"
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/test_python.sh cpu
    """
    deleteDir()
  }
}

def TestPythonGPU(args) {
  def nodeReq = (args.multi_gpu) ? 'linux && mgpu' : 'linux && gpu'
  def artifact_cuda_version = (args.artifact_cuda_version) ?: ref_cuda_ver
  node(nodeReq) {
    unstash name: "xgboost_whl_cuda${artifact_cuda_version}"
    unstash name: "xgboost_cpp_tests_cuda${artifact_cuda_version}"
    unstash name: 'srcs'
    echo "Test Python GPU: CUDA ${args.host_cuda_version}"
    def container_type = "gpu"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.host_cuda_version}"
    def mgpu_indicator = (args.multi_gpu) ? 'mgpu' : 'gpu'
    // Allocate extra space in /dev/shm to enable NCCL
    def docker_extra_params = (args.multi_gpu) ? "CI_DOCKER_EXTRA_PARAMS_INIT='--shm-size=4g'" : ''
    sh "${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_python.sh ${mgpu_indicator}"
    if (args.test_rmm) {
      sh "rm -rfv build/ python-package/dist/"
      unstash name: "xgboost_whl_rmm_cuda${args.host_cuda_version}"
      unstash name: "xgboost_cpp_tests_rmm_cuda${args.host_cuda_version}"
      sh "${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_python.sh ${mgpu_indicator} --use-rmm-pool"
    }
    deleteDir()
  }
}

def TestCppRabit() {
  node(nodeReq) {
    unstash name: 'xgboost_rabit_tests'
    unstash name: 'srcs'
    echo "Test C++, rabit mock on"
    def container_type = "cpu"
    def docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/runxgb.sh xgboost tests/ci_build/approx.conf.in
    """
    deleteDir()
  }
}

def TestCppGPU(args) {
  def nodeReq = 'linux && mgpu'
  def artifact_cuda_version = (args.artifact_cuda_version) ?: ref_cuda_ver
  node(nodeReq) {
    unstash name: "xgboost_cpp_tests_cuda${artifact_cuda_version}"
    unstash name: 'srcs'
    echo "Test C++, CUDA ${args.host_cuda_version}"
    def container_type = "gpu"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.host_cuda_version}"
    sh "${dockerRun} ${container_type} ${docker_binary} ${docker_args} build/testxgboost"
    if (args.test_rmm) {
      sh "rm -rfv build/"
      unstash name: "xgboost_cpp_tests_rmm_cuda${args.host_cuda_version}"
      echo "Test C++, CUDA ${args.host_cuda_version} with RMM"
      container_type = "rmm"
      docker_binary = "nvidia-docker"
      docker_args = "--build-arg CUDA_VERSION=${args.host_cuda_version}"
      sh """
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} bash -c "source activate gpu_test && build/testxgboost --use-rmm-pool --gtest_filter=-*DeathTest.*"
      """
    }
    deleteDir()
  }
}

def CrossTestJVMwithJDKGPU(args) {
  def nodeReq = 'linux && mgpu'
  node(nodeReq) {
    unstash name: "xgboost4j_jar_gpu"
    unstash name: 'srcs'
    if (args.spark_version != null) {
      echo "Test XGBoost4J on a machine with JDK ${args.jdk_version}, Spark ${args.spark_version}, CUDA ${args.host_cuda_version}"
    } else {
      echo "Test XGBoost4J on a machine with JDK ${args.jdk_version}, CUDA ${args.host_cuda_version}"
    }
    def container_type = "gpu_jvm"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.host_cuda_version}"
    sh "${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_jvm_gpu_cross.sh"
    deleteDir()
  }
}

def CrossTestJVMwithJDK(args) {
  node('linux && cpu') {
    unstash name: 'xgboost4j_jar'
    unstash name: 'srcs'
    if (args.spark_version != null) {
      echo "Test XGBoost4J on a machine with JDK ${args.jdk_version}, Spark ${args.spark_version}"
    } else {
      echo "Test XGBoost4J on a machine with JDK ${args.jdk_version}"
    }
    def container_type = "jvm_cross"
    def docker_binary = "docker"
    def spark_arg = (args.spark_version != null) ? "--build-arg SPARK_VERSION=${args.spark_version}" : ""
    def docker_args = "--build-arg JDK_VERSION=${args.jdk_version} ${spark_arg}"
    // Run integration tests only when spark_version is given
    def docker_extra_params = (args.spark_version != null) ? "CI_DOCKER_EXTRA_PARAMS_INIT='-e RUN_INTEGRATION_TEST=1'" : ""
    sh """
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_jvm_cross.sh
    """
    deleteDir()
  }
}

def TestR(args) {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Test R package"
    def container_type = "rproject"
    def docker_binary = "docker"
    def use_r35_flag = (args.use_r35) ? "1" : "0"
    def docker_args = "--build-arg USE_R35=${use_r35_flag}"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_test_rpkg.sh || tests/ci_build/print_r_stacktrace.sh
    """
    deleteDir()
  }
}

def DeployJVMPackages(args) {
  node('linux && cpu') {
    unstash name: 'srcs'
    if (env.BRANCH_NAME == 'master' || env.BRANCH_NAME.startsWith('release')) {
      echo 'Deploying to xgboost-maven-repo S3 repo...'
      def container_type = "jvm"
      def docker_binary = "docker"
      sh """
      ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/deploy_jvm_packages.sh ${args.spark_version}
      """
    }
    deleteDir()
  }
}
