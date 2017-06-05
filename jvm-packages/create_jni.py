#!/usr/bin/env python
import errno
import glob
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from subprocess import check_output


#: Enable of your compiler supports OpenMP.
USE_OMP = "ON"


@contextmanager
def cd(path):
    path = normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def run(command, **kwargs):
    print(command)
    subprocess.check_call(command, shell=True, **kwargs)


def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target))
    shutil.copy(source, target)


def normpath(path):
    """Normalize UNIX path to a native path."""
    return os.path.join(*path.split("/"))


if __name__ == "__main__":
    if sys.platform == "darwin":
        USE_OMP = "OFF"
        os.environ["JAVA_HOME"] = check_output(
            "/usr/libexec/java_home").strip().decode()

    print("building Java wrapper")
    with cd(".."):
        maybe_makedirs("build")
        with cd("build"):
            if sys.platform == "win32":
                # Force x64 build on Windows.
                maybe_generator = '-G"Visual Studio 14 Win64"'
            else:
                maybe_generator = ""

            run("cmake .. -DJVM_BINDINGS:BOOL=ON -DUSE_OPENMP:BOOL={0} {1}"
                .format(USE_OMP, maybe_generator))
            run("cmake --build .")

        with cd("demo/regression"):
            run(sys.executable + " mapfeat.py")
            run(sys.executable + " mknfold.py machine.txt 1")

    print("copying native library")
    library_name = {
        "win32": "xgboost4j.dll",
        "darwin": "libxgboost4j.dylib",
        "linux2": "libxgboost4j.so"
    }[sys.platform]
    maybe_makedirs("xgboost4j/src/main/resources/lib")
    cp("../lib/" + library_name, "xgboost4j/src/main/resources/lib")

    print("copying pure-Python tracker")
    cp("../dmlc-core/tracker/dmlc_tracker/tracker.py",
       "xgboost4j/src/main/resources")

    print("copying train/test files")
    maybe_makedirs("xgboost4j-spark/src/test/resources")
    with cd("../demo/regression"):
        run("{} mapfeat.py".format(sys.executable))
        run("{} mknfold.py machine.txt 1".format(sys.executable))

    for file in glob.glob("../demo/regression/machine.txt.t*"):
        cp(file, "xgboost4j-spark/src/test/resources")
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "xgboost4j-spark/src/test/resources")
