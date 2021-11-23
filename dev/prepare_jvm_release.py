import os
import sys
import errno
import subprocess
import glob
import shutil
from contextlib import contextmanager

def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    else:
        return normalized

def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target))
    shutil.copy(source, target)

def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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

def run(command, **kwargs):
    print(command)
    subprocess.check_call(command, shell=True, **kwargs)

def main():
    with cd("jvm-packages/"):
        print("====copying pure-Python tracker====")
        for use_cuda in [True, False]:
            xgboost4j = "xgboost4j-gpu" if use_cuda else "xgboost4j"
            cp("../python-package/xgboost/tracker.py", f"{xgboost4j}/src/main/resources")

        print("====copying resources for testing====")
        with cd("../demo/CLI/regression"):
            run(f"{sys.executable} mapfeat.py")
            run(f"{sys.executable} mknfold.py machine.txt 1")
        for use_cuda in [True, False]:
            xgboost4j = "xgboost4j-gpu" if use_cuda else "xgboost4j"
            xgboost4j_spark = "xgboost4j-spark-gpu" if use_cuda else "xgboost4j-spark"
            maybe_makedirs(f"{xgboost4j}/src/test/resources")
            maybe_makedirs(f"{xgboost4j_spark}/src/test/resources")
            for file in glob.glob("../demo/data/agaricus.*"):
                cp(file, f"{xgboost4j}/src/test/resources")
                cp(file, f"{xgboost4j_spark}/src/test/resources")
            for file in glob.glob("../demo/CLI/regression/machine.txt.t*"):
                cp(file, f"{xgboost4j_spark}/src/test/resources")

        print("====Creating directories to hold native binaries====")
        for os, arch in [("linux", "x86_64"), ("windows", "x86_64"), ("macos", "x86_64")]:
            output_dir = f"xgboost4j/src/main/resources/lib/{os}/{arch}"
            maybe_makedirs(output_dir)
        for os, arch in [("linux", "x86_64")]:
            output_dir = f"xgboost4j-gpu/src/main/resources/lib/{os}/{arch}"
            maybe_makedirs(output_dir)
    print("====Next Steps====")
    print("1. Obtain Linux and Windows binaries from the CI server")
    print("2. Put them in xgboost4j(-gpu)/src/main/resources/lib/[os]/[arch]")
    print("3. Now on a Mac machine, run:")
    print("   GPG_TTY=$(tty) mvn deploy -Prelease -DskipTests")

if __name__ == "__main__":
    main()
