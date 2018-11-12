#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

import groovy.transform.Field

/* Unrestricted tasks: tasks that do NOT generate artifacts */

// Command to run command inside a docker container
def dockerRun = 'tests/ci_build/ci_build.sh'
// Utility functions
@Field
def utils

def buildMatrix = [
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": true,  "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "9.2", "multiGpu": true],
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": true,  "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "9.2" ],
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": true,  "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "8.0" ],
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": false, "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "8.0" ],
]

pipeline {
    // Each stage specify its own agent
    agent none

    // Setup common job properties
    options {
        ansiColor('xterm')
        timestamps()
        timeout(time: 120, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    // Build stages
    stages {
        stage('Jenkins: Get sources') {
            agent {
                label 'unrestricted'
            }
            steps {
                script {
                    utils = load('tests/ci_build/jenkins_tools.Groovy')
                    utils.checkoutSrcs()
                }
                stash name: 'srcs', excludes: '.git/'
                milestone label: 'Sources ready', ordinal: 1
            }
        }
        stage('Jenkins: Build & Test') {
            steps {
                script {
                    parallel (buildMatrix.findAll{it['enabled']}.collectEntries{ c ->
                        def buildName = utils.getBuildName(c)
                        utils.buildFactory(buildName, c, false, this.&buildPlatformCmake)
                    })
                }
            }
        }
    }
}

/**
 * Build platform and test it via cmake.
 */
def buildPlatformCmake(buildName, conf, nodeReq, dockerTarget) {
    def opts = utils.cmakeOptions(conf)
    // Destination dir for artifacts
    def distDir = "dist/${buildName}"
    def dockerArgs = ""
    if (conf["withGpu"]) {
        dockerArgs = "--build-arg CUDA_VERSION=" + conf["cudaVersion"]
    }
    def test_suite = conf["withGpu"] ? (conf["multiGpu"] ? "mgpu" : "gpu") : "cpu"
    // Build node - this is returned result
    retry(3) {
        node(nodeReq) {
            unstash name: 'srcs'
            echo """
            |===== XGBoost CMake build =====
            |  dockerTarget: ${dockerTarget}
            |  cmakeOpts   : ${opts}
            |=========================
            """.stripMargin('|')
            // Invoke command inside docker
            sh """
            ${dockerRun} ${dockerTarget} ${dockerArgs} tests/ci_build/build_via_cmake.sh ${opts}
            ${dockerRun} ${dockerTarget} ${dockerArgs} tests/ci_build/test_${test_suite}.sh
            """
            if (!conf["multiGpu"]) {
                sh """
                ${dockerRun} ${dockerTarget} ${dockerArgs} bash -c "cd python-package; rm -f dist/*; python setup.py bdist_wheel --universal"
                rm -rf "${distDir}"; mkdir -p "${distDir}/py"
                cp xgboost "${distDir}"
                cp -r python-package/dist "${distDir}/py"
                # Test the wheel for compatibility on a barebones CPU container
                ${dockerRun} release ${dockerArgs} bash -c " \
                    pip install --user python-package/dist/xgboost-*-none-any.whl && \
                    python -m nose -v tests/python"
                # Test the wheel for compatibility on CUDA 10.0 container
                ${dockerRun} gpu --build-arg CUDA_VERSION=10.0 bash -c " \
                    pip install --user python-package/dist/xgboost-*-none-any.whl && \
                    python -m nose -v --eval-attr='(not slow) and (not mgpu)' tests/python-gpu"
                """
            }
        }
    }
}
