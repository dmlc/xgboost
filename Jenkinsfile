#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build.sh'

pipeline {
  // Each stage specify its own agent
  agent none

  environment {
    DOCKER_CACHE_REPO = '492475357299.dkr.ecr.us-west-2.amazonaws.com'
  }

  // Setup common job properties
  options {
    ansiColor('xterm')
    timestamps()
    timeout(time: 120, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '10'))
    preserveStashes()
  }

  // Build stages
  stages {
    stage('Get sources') {
      agent { label 'linux && cpu' }
      steps {
        script {
          checkoutSrcs()
        }
        stash name: 'srcs'
        milestone ordinal: 1
      }
    }
    stage('Formatting Check') {
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
    stage('Build') {
      agent none
      steps {
        script {
          parallel ([
            'build-cpu': { BuildCPU() },
            'build-gpu-cuda8.0': { BuildCUDA(cuda_version: '8.0') },
            'build-gpu-cuda9.2': { BuildCUDA(cuda_version: '9.2') },
            'build-gpu-cuda10.0': { BuildCUDA(cuda_version: '10.0') },
            'build-jvm-packages': { BuildJVMPackages(spark_version: '2.4.1') },
            'build-jvm-doc': { BuildJVMDoc() }
          ])
        }
        milestone ordinal: 3
      }
    }
    stage('Test') {
      agent none
      steps {
        script {
          parallel ([
            'test-python-cpu': { TestPythonCPU() },
            'test-python-gpu-cuda8.0': { TestPythonGPU(cuda_version: '8.0') },
            'test-python-gpu-cuda9.2': { TestPythonGPU(cuda_version: '9.2') },
            'test-python-gpu-cuda10.0': { TestPythonGPU(cuda_version: '10.0') },
            'test-cpp-gpu': { TestCppGPU(cuda_version: '10.0') },
            'test-cpp-mgpu': { TestCppGPU(cuda_version: '10.0', multi_gpu: true) },
            'test-jvm-jdk8': { CrossTestJVMwithJDK(jdk_version: '8') },
            'test-jvm-jdk11': { CrossTestJVMwithJDK(jdk_version: '11') },
            'test-r-3.4.4': { TestR(r_version: '3.4.4') },
            'test-r-3.5.3': { TestR(r_version: '3.5.3') }
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
    container_type = "clang_tidy"
    docker_binary = "docker"
    dockerArgs = "--build-arg CUDA_VERSION=9.2"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${dockerArgs} tests/ci_build/clang_tidy.sh
    """
    deleteDir()
  }
}

def Lint() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Running lint..."
    // commented out for now, until another PR to migrate lint to Python 3 gets merged
    container_type = "lint"
    docker_binary = "docker"
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
    container_type = "lint"
    docker_binary = "docker"
    docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='-e SPHINX_GIT_BRANCH=${BRANCH_NAME}'"
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
    container_type = "lint"
    docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/doxygen.sh
    """
    deleteDir()
  }
}

def BuildCPU() {
  node('linux && cpu') {
    echo "Build CPU"
  }
}

def BuildCUDA(args) {
  node('linux && cpu') {
    echo "Build with CUDA ${args.cuda_version}"
  }
}

def BuildJVMPackages(args) {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Build XGBoost4J-Spark with Spark ${args.spark_version}"
    container_type = "jvm"
    docker_binary = "docker"
    // Use only 4 CPU cores
    docker_extra_params = "CI_DOCKER_EXTRA_PARAMS_INIT='--cpuset-cpus 0-3'"
    sh """
    ${docker_extra_params} ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_jvm_packages.sh
    """
    echo 'Stashing XGBoost4J JAR...'
    stash name: 'xgboost4j_jar', includes: 'jvm-packages/xgboost4j/target/*.jar'
    deleteDir()
  }
}

def BuildJVMDoc() {
  node('linux && cpu') {
    unstash name: 'srcs'
    echo "Building JVM doc..."
    container_type = "jvm"
    docker_binary = "docker"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} tests/ci_build/build_jvm_doc.sh ${BRANCH_NAME}
    """
    archiveArtifacts artifacts: "jvm-packages/${BRANCH_NAME}.tar.bz2", allowEmptyArchive: true
    echo 'Uploading doc...'
    s3Upload file: "jvm-packages/${BRANCH_NAME}.tar.bz2", bucket: 'xgboost-docs', acl: 'PublicRead', path: "${BRANCH_NAME}.tar.bz2"
    deleteDir()
  }
}

def TestPythonCPU() {
  node('linux && cpu') {
    echo "Test Python CPU"
  }
}

def TestPythonGPU(args) {
  node('linux && cpu') {
    echo "Test Python GPU: CUDA ${args.cuda_version}"
  }
}

def TestCppGPU(args) {
  node('linux && cpu') {
    echo "Test C++, CUDA ${args.cuda_version}"
    if (args.multi_gpu) {
      echo "Using multiple GPUs"
    }
  }
}

def CrossTestJVMwithJDK(args) {
  node('linux && cpu') {
    unstash name: 'xgboost4j_jar'
    echo "Test XGBoost4J on a machine with JDK ${args.jdk_version}"
    container_type = "jvm_cross"
    docker_binary = "docker"
    docker_args = "--build-arg JDK_VERSION=${args.jdk_version}"
    sh """
    ${dockerRun} ${container_type} ${docker_binary} ${docker_args} tests/ci_build/test_jvm_cross.sh
    """
    deleteDir()
  }
}

def TestR(args) {
  node('linux && cpu') {
    echo "Test R package: R version ${args.r_version}"
  }
}
