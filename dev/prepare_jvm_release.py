import argparse
import errno
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from urllib.request import urlretrieve


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

def get_current_git_tag():
    out = subprocess.check_output(["git", "tag", "--points-at", "HEAD"])
    return out.decode().split("\n")[0]

def get_current_commit_hash():
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode().split("\n")[0]

def get_current_git_branch():
    out = subprocess.check_output(["git", "log", "-n", "1", "--pretty=%d", "HEAD"])
    m = re.search(r"release_[0-9\.]+", out.decode())
    if not m:
        raise ValueError("Expected branch name of form release_xxx")
    return m.group(0)

def retrieve(url, filename=None):
    print(f"{url} -> {filename}")
    return urlretrieve(url, filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-version", type=str, required=True,
                        help="Version of the release being prepared")
    args = parser.parse_args()

    if sys.platform != "darwin" or platform.machine() != "x86_64":
        raise NotImplementedError("Please run this script using an Intel Mac")

    version = args.release_version
    expected_git_tag = "v" + version
    current_git_tag = get_current_git_tag()
    if current_git_tag != expected_git_tag:
        if not current_git_tag:
            raise ValueError(f"Expected git tag {expected_git_tag} but current HEAD has no tag. "
                             f"Run: git checkout {expected_git_tag}")
        raise ValueError(f"Expected git tag {expected_git_tag} but current HEAD is at tag "
                         f"{current_git_tag}. Run: git checkout {expected_git_tag}")

    commit_hash = get_current_commit_hash()
    git_branch = get_current_git_branch()
    print(f"Using commit {commit_hash} of branch {git_branch}, git tag {current_git_tag}")

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
        for os_ident, arch in [("linux", "x86_64"), ("windows", "x86_64"), ("macos", "x86_64")]:
            output_dir = f"xgboost4j/src/main/resources/lib/{os_ident}/{arch}"
            maybe_makedirs(output_dir)
        for os_ident, arch in [("linux", "x86_64")]:
            output_dir = f"xgboost4j-gpu/src/main/resources/lib/{os_ident}/{arch}"
            maybe_makedirs(output_dir)

        print("====Downloading native binaries from CI====")
        nightly_bucket_prefix = "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds"
        maven_repo_prefix = "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/release/ml/dmlc"

        retrieve(url=f"{nightly_bucket_prefix}/{git_branch}/xgboost4j_{commit_hash}.dll",
                 filename="xgboost4j/src/main/resources/lib/windows/x86_64/xgboost4j.dll")

        with tempfile.TemporaryDirectory() as tempdir:
            # libxgboost4j.so for Linux x86_64, CPU only
            zip_path = os.path.join(tempdir, "xgboost4j_2.12.jar")
            extract_dir = os.path.join(tempdir, "xgboost4j")
            retrieve(url=f"{maven_repo_prefix}/xgboost4j_2.12/{version}/"
                         f"xgboost4j_2.12-{version}.jar",
                     filename=zip_path)
            os.mkdir(extract_dir)
            with zipfile.ZipFile(zip_path, "r") as t:
                t.extractall(extract_dir)
            cp(os.path.join(extract_dir, "lib", "linux", "x86_64", "libxgboost4j.so"),
               "xgboost4j/src/main/resources/lib/linux/x86_64/libxgboost4j.so")

            # libxgboost4j.so for Linux x86_64, GPU support
            zip_path = os.path.join(tempdir, "xgboost4j-gpu_2.12.jar")
            extract_dir = os.path.join(tempdir, "xgboost4j-gpu")
            retrieve(url=f"{maven_repo_prefix}/xgboost4j-gpu_2.12/{version}/"
                         f"xgboost4j-gpu_2.12-{version}.jar",
                     filename=zip_path)
            os.mkdir(extract_dir)
            with zipfile.ZipFile(zip_path, "r") as t:
                t.extractall(extract_dir)
            cp(os.path.join(extract_dir, "lib", "linux", "x86_64", "libxgboost4j.so"),
               "xgboost4j-gpu/src/main/resources/lib/linux/x86_64/libxgboost4j.so")


    print("====Next Steps====")
    print("1. Gain upload right to Maven Central repo.")
    print("1-1. Sign up for a JIRA account at Sonatype: ")
    print("1-2. File a JIRA ticket: "
          "https://issues.sonatype.org/secure/CreateIssue.jspa?issuetype=21&pid=10134. Example: "
          "https://issues.sonatype.org/browse/OSSRH-67724")
    print("2. Store the Sonatype credentials in .m2/settings.xml. See insturctions in "
          "https://central.sonatype.org/publish/publish-maven/")
    print("3. Now on a Mac machine, run:")
    print("   GPG_TTY=$(tty) mvn deploy -Prelease -DskipTests")
    print("4. Log into https://oss.sonatype.org/. On the left menu panel, click Staging "
          "Repositories. Visit the URL https://oss.sonatype.org/content/repositories/mldmlc-1085 "
          "to inspect the staged JAR files. Finally, press Release button to publish the "
          "artifacts to the Maven Central repository.")

if __name__ == "__main__":
    main()
