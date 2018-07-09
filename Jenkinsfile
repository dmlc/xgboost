#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build.sh'

def buildMatrix = [
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": true,  "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "9.2" ],
    [ "enabled": true,  "os" : "linux", "withGpu": true, "withNccl": true,  "withOmp": true, "pythonVersion": "2.7", "cudaVersion": "8.0" ],
    [ "enabled": false,  "os" : "linux", "withGpu": false, "withNccl": false, "withOmp": true, "pythonVersion": "2.7", "cudaVersion": ""  ],
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
        stage('Get sources') {
            agent any
            steps {
                checkoutSrcs()
                stash name: 'srcs', excludes: '.git/'
                milestone label: 'Sources ready', ordinal: 1
            }
        }
        stage('Build & Test') {
            steps {
                script {
                    parallel (buildMatrix.findAll{it['enabled']}.collectEntries{ c ->
                        def buildName = getBuildName(c)
                        buildFactory(buildName, c)
                    })
                }
            }
        }
    }
}

// initialize source codes
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

/**
 * Creates cmake and make builds
 */
def buildFactory(buildName, conf) {
    def os = conf["os"]
    def nodeReq = conf["withGpu"] ? "${os} && gpu" : "${os}"
    def dockerTarget = conf["withGpu"] ? "gpu" : "cpu"
    [ ("${buildName}") : { buildPlatformCmake("${buildName}", conf, nodeReq, dockerTarget) }
    ]
}

/**
 * Build platform and test it via cmake.
 */
def buildPlatformCmake(buildName, conf, nodeReq, dockerTarget) {
    def opts = cmakeOptions(conf)
    // Destination dir for artifacts
    def distDir = "dist/${buildName}"
    def dockerArgs = ""
    if(conf["withGpu"]){
        dockerArgs = "--build-arg CUDA_VERSION=" + conf["cudaVersion"]
    }
    // Build node - this is returned result
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
        ${dockerRun} ${dockerTarget} ${dockerArgs} tests/ci_build/test_${dockerTarget}.sh
        ${dockerRun} ${dockerTarget} ${dockerArgs} bash -c "cd python-package; rm -f dist/*; python setup.py bdist_wheel --universal"
        rm -rf "${distDir}"; mkdir -p "${distDir}/py"
        cp xgboost "${distDir}"
        cp -r lib "${distDir}"
        cp -r python-package/dist "${distDir}/py"
        # Test the wheel for compatibility on a barebones CPU container
        ${dockerRun} release ${dockerArgs} bash -c " \
            auditwheel show xgboost-*-py2-none-any.whl
            pip install --user python-package/dist/xgboost-*-none-any.whl && \
            python -m nose tests/python"
        """
        archiveArtifacts artifacts: "${distDir}/**/*.*", allowEmptyArchive: true
    }
}

def cmakeOptions(conf) {
    return ([
        conf["withGpu"] ? '-DUSE_CUDA=ON' : '-DUSE_CUDA=OFF',
        conf["withNccl"] ? '-DUSE_NCCL=ON' : '-DUSE_NCCL=OFF',
        conf["withOmp"] ? '-DOPEN_MP:BOOL=ON' : '']
        ).join(" ")
}

def getBuildName(conf) {
    def gpuLabel = conf['withGpu'] ? "_cuda" + conf['cudaVersion'] : "_cpu"
    def ompLabel = conf['withOmp'] ? "_omp" : ""
    def pyLabel = "_py${conf['pythonVersion']}"
    return "${conf['os']}${gpuLabel}${ompLabel}${pyLabel}"
}

