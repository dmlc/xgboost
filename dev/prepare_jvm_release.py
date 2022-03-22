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
    print("1. Gain upload right to Maven Central repo.")
    print("1-1. Sign up for a JIRA account at Sonatype: ")
    print("1-2. File a JIRA ticket: "
          "https://issues.sonatype.org/secure/CreateIssue.jspa?issuetype=21&pid=10134. Example: "
          "https://issues.sonatype.org/browse/OSSRH-67724")
    print("2. Store the Sonatype credentials in .m2/settings.xml. See insturctions in "
          "https://central.sonatype.org/publish/publish-maven/")
    print("3. Obtain Linux and Windows binaries from the CI server")
    print("3-1. Get xgboost4j_[commit].dll from "
          "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html. Rename it to"
          "xgboost4j.dll.")
    print("3-2. For Linux binaries, go to "
          "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/list.html and navigate to the "
          "release/ directory. Find and download two JAR files: xgboost4j_2.12-[version].jar and "
          "xgboost4j-gpu_2.12-[version].jar. Use unzip command to extract libxgboost4j.so (one "
          "version compiled with GPU support and another compiled without).")
    print("4. Put the binaries in xgboost4j(-gpu)/src/main/resources/lib/[os]/[arch]")
    print("5. Now on a Mac machine, run:")
    print("   GPG_TTY=$(tty) mvn deploy -Prelease -DskipTests")
    print("6. Log into https://oss.sonatype.org/. On the left menu panel, click Staging "
          "Repositories. Visit the URL https://oss.sonatype.org/content/repositories/mldmlc-1085 "
          "to inspect the staged JAR files. Finally, press Release button to publish the "
          "artifacts to the Maven Central repository.")

if __name__ == "__main__":
    main()
