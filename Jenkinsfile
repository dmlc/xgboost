// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'

// timeout in minutes
max_time = 60

// initialize source codes
def init_git() {
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

stage('Build') {
    node('GPU' && 'linux') {
      ws('workspace/xgboost/build-gpu-cmake') {
        init_git()
  	timeout(time: max_time, unit: 'MINUTES') {
      		sh "${docker_run} gpu tests/ci_build/build_gpu_cmake.sh"
	}
      }
    }
    node('GPU' && 'linux') {
      ws('workspace/xgboost/build-gpu-make') {
        init_git()
  	timeout(time: max_time, unit: 'MINUTES') {
      		sh "${docker_run} gpu make PLUGIN_UPDATER_GPU=ON"
	}
      }
    }
}


stage('Unit Test') {
    node('GPU' && 'linux') {
      ws('workspace/xgboost/unit-test') {
        init_git()
  	timeout(time: max_time, unit: 'MINUTES') {
      		sh "${docker_run} gpu tests/ci_build/test_gpu.ssh"
	}
      }
    }
}
