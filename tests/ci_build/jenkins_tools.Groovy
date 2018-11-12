#!/usr/bin/groovy
// -*- mode: groovy -*-

/* Utility functions for Jenkins */

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build.sh'

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
def buildFactory(buildName, conf, restricted, build_func) {
    def os = conf["os"]
    def device = conf["withGpu"] ? (conf["multiGpu"] ? "mgpu" : "gpu") : "cpu"
    def restricted_flag = restricted ? "restricted" : "unrestricted"
    def nodeReq = "${os} && ${device} && ${restricted_flag}"
    def dockerTarget = conf["withGpu"] ? "gpu" : "cpu"
    [ ("${buildName}") : { build_func("${buildName}", conf, nodeReq, dockerTarget) }
    ]
}

def cmakeOptions(conf) {
    return ([
        conf["withGpu"] ? '-DUSE_CUDA=ON' : '-DUSE_CUDA=OFF',
        conf["withNccl"] ? '-DUSE_NCCL=ON' : '-DUSE_NCCL=OFF',
        conf["withOmp"] ? '-DOPEN_MP:BOOL=ON' : '']
        ).join(" ")
}

def getBuildName(conf) {
    def gpuLabel = conf['withGpu'] ? ( (conf['multiGpu'] ? "_mgpu" : "") + "_cuda" + conf['cudaVersion'] + (conf['withNccl'] ? "_nccl" : "_nonccl")) : "_cpu"
    def ompLabel = conf['withOmp'] ? "_omp" : ""
    def pyLabel = "_py${conf['pythonVersion']}"
    return "${conf['os']}${gpuLabel}${ompLabel}${pyLabel}"
}

return this
