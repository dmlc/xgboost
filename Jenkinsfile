#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build.sh'

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
    stage('Jenkins Linux: Get sources') {
      agent { label 'linux && cpu' }
      steps {
        script {
          checkoutSrcs()
          commit_id = "${GIT_COMMIT}"
        }
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
            'build-gpu-cuda9.0': { BuildCUDA(cuda_version: '9.0') },
            'build-gpu-cuda10.0': { BuildCUDA(cuda_version: '10.0') },
            'build-gpu-cuda10.1': { BuildCUDA(cuda_version: '10.1') },
            'build-jvm-packages': { BuildJVMPackages(spark_version: '2.4.3') },
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
            'test-python-gpu-cuda9.0': { TestPythonGPU(cuda_version: '9.0') },
            'test-python-gpu-cuda10.0': { TestPythonGPU(cuda_version: '10.0') },
            'test-python-gpu-cuda10.1': { TestPythonGPU(cuda_version: '10.1') },
            'test-python-mgpu-cuda10.1': { TestPythonGPU(cuda_version: '10.1', multi_gpu: true) },
            'test-cpp-gpu': { TestCppGPU(cuda_version: '10.1') },
            'test-cpp-mgpu': { TestCppGPU(cuda_version: '10.1', multi_gpu: true) },
            'test-jvm-jdk8': { CrossTestJVMwithJDK(jdk_version: '8', spark_version: '2.4.3') },
            'test-jvm-jdk11': { CrossTestJVMwithJDK(jdk_version: '11') },
            'test-jvm-jdk12': { CrossTestJVMwithJDK(jdk_version: '12') },
            'test-r-3.4.4': { TestR(use_r35: false) },
            'test-r-3.5.3': { TestR(use_r35: true) }
          ])
        }
        milestone ordinal: 4
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

def ClangTidy() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Running clang-tidy job..."
    def container_type = "clang_tidy"
    def docker_binary = "docker"
    def dockerArgs = "--build-arg CUDA_VERSION=9.2"
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
    ${dockerRun} ${container_type} ${docker_binary} make lint
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
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} make -C doc html
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
    echo 'Uploading doc...'
    s3Upload file: "build/${BRANCH_NAME}.tar.bz2", bucket: 'xgboost-docs', acl: 'PublicRead', path: "doxygen/${BRANCH_NAME}.tar.bz2"
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
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_via_cmake.sh
    ${dockerRun} ${container_type} ${docker_binary} build/testxgboost
    """
    // Sanitizer test
    def docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='-e ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer -e ASAN_OPTIONS=symbolize=1 --cap-add SYS_PTRACE'"
    def docker_args = "--build-arg CMAKE_VERSION=3.12"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_via_cmake.sh -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address" \
      -DCMAKE_BUILD_TYPE=Debug -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} build/testxgboost
    """
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


def BuildCUDA(args) {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build with CUDA ${args.cuda_version}"
    def container_type = "gpu_build"
    def docker_binary = "docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_NCCL=ON -DOPEN_MP:BOOL=ON
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} bash -c "cd python-package && rm -rf dist/* && python setup.py bdist_wheel --universal"
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} python3 tests/ci_build/rename_whl.py python-package/dist/*.whl ${commit_id} manylinux1_x86_64
    """
    // Stash wheel for CUDA 9.0 target
    if (args.cuda_version == '9.0') {
      echo 'Stashing Python wheel...'
      stash name: 'xgboost_whl_cuda9', includes: 'python-package/dist/*.whl'
      path = ("${BRANCH_NAME}" == 'master') ? '' : "${BRANCH_NAME}/"
      s3Upload bucket: 'xgboost-nightly-builds', path: path, acl: 'PublicRead', workingDir: 'python-package/dist', includePathPattern:'**/*.whl'
      echo 'Stashing C++ test executable (testxgboost)...'
      stash name: 'xgboost_cpp_tests', includes: 'build/testxgboost'
    }
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
    stash name: 'xgboost4j_jar', includes: 'jvm-packages/xgboost4j/target/*.jar,jvm-packages/xgboost4j-spark/target/*.jar,jvm-packages/xgboost4j-example/target/*.jar'
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
    echo 'Uploading doc...'
    s3Upload file: "jvm-packages/${BRANCH_NAME}.tar.bz2", bucket: 'xgboost-docs', acl: 'PublicRead', path: "${BRANCH_NAME}.tar.bz2"
    deleteDir()
  }
}

def TestPythonCPU() {
  node('linux && cpu') {
    unstash name: 'xgboost_whl_cuda9'
    unstash name: 'srcs'
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
  nodeReq = (args.multi_gpu) ? 'linux && mgpu' : 'linux && gpu'
  node(nodeReq) {
    unstash name: 'xgboost_whl_cuda9'
    unstash name: 'srcs'
    echo "Test Python GPU: CUDA ${args.cuda_version}"
    def container_type = "gpu"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
    if (args.multi_gpu) {
      echo "Using multiple GPUs"
      sh """
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_python.sh mgpu
      """
    } else {
      echo "Using a single GPU"
      sh """
      ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_python.sh gpu
      """
    }
    // For CUDA 10.0 target, run cuDF tests too
    if (args.cuda_version == '10.0') {
      echo "Running tests with cuDF..."
      sh """
      ${dockerRun} cudf ${docker_binary} ${docker_args} tests/ci_build/test_python.sh cudf
      """
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
  nodeReq = (args.multi_gpu) ? 'linux && mgpu' : 'linux && gpu'
  node(nodeReq) {
    unstash name: 'xgboost_cpp_tests'
    unstash name: 'srcs'
    echo "Test C++, CUDA ${args.cuda_version}"
    def container_type = "gpu"
    def docker_binary = "nvidia-docker"
    def docker_args = "--build-arg CUDA_VERSION=${args.cuda_version}"
    if (args.multi_gpu) {
      echo "Using multiple GPUs"
      sh "${dockerRun} ${container_type} ${docker_binary} ${docker_args} build/testxgboost --gtest_filter=*.MGPU_*"
    } else {
      echo "Using a single GPU"
      sh "${dockerRun} ${container_type} ${docker_binary} ${docker_args} build/testxgboost --gtest_filter=-*.MGPU_*"
    }
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
