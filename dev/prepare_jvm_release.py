"""
Helper script to prepare for releasing XGBoost JVM packages to
Maven Central.

## Prerequisite

1. You must have the right to upload artifacts to the Maven Central repo.
   If you do not, contact Hyunsu Cho (chohyu01@cs.washington.edu) so that
   he can contact Sonatype on your behalf in order to add you as a
   "producer" user for the ml.dmlc namespace. See
   https://central.sonatype.org/pages/support/#status to learn about
   the process for adding or removing users who can publish to the project.

2. Follow instructions in
   https://central.sonatype.org/publish/publish-portal-maven/#credentials
   to set up the authentication token in your machine.

3. Set up GPG for signing artifacts:
   https://central.sonatype.org/publish/requirements/gpg/

## Making the release

Run this script 4 times:

python3 dev/prepare_jvm_release.py --scala-version 2.12 --variant cpu --release=x.x.x
python3 dev/prepare_jvm_release.py --scala-version 2.12 --variant gpu --release=x.x.x
python3 dev/prepare_jvm_release.py --scala-version 2.13 --variant cpu --release=x.x.x
python3 dev/prepare_jvm_release.py --scala-version 2.13 --variant gpu --release=x.x.x

"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import chdir
from typing import Literal
from urllib.request import urlretrieve


def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    return normalized


def cp(source, target):
    """Copy a file after normalizing both paths."""
    source = normpath(source)
    target = normpath(target)
    print(f"cp {source} {target}")
    shutil.copy(source, target)


def maybe_makedirs(path):
    """Create a directory and its parents if needed."""
    path = normpath(path)
    print("mkdir -p " + path)
    os.makedirs(path, exist_ok=True)


def run(command, **kwargs):
    """Run a shell command and fail if it exits with an error."""
    print(command)
    subprocess.run(command, shell=True, check=True, **kwargs)


def deploy(local: bool, profile: Literal["default", "gpu"], pl: str | None) -> None:
    """Deploy JVM artifacts with the selected Maven profile."""
    subcmd = "install" if local else "deploy"

    cmd = (
        f"mvn {subcmd} -P{profile},release "
        "-DskipTests -Dmaven.test.skip=true -Dskip.native.build=true"
    )
    if pl:
        cmd += f" -pl {pl}"

    run(cmd)


def get_current_commit_hash() -> str:
    """Get the last commit of the current branch."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode().split("\n", maxsplit=1)[0]


def get_current_git_branch() -> str:
    """Get the current branch."""
    out = subprocess.check_output(["git", "log", "-n", "1", "--pretty=%d", "HEAD"])
    m = re.search(r"release_[0-9\.]+", out.decode())
    if not m:
        raise ValueError("Expected branch name of form release_xxx")
    return m.group(0)


def retrieve(url, filename=None):
    """Download a file from a URL and print the destination."""
    print(f"{url} -> {filename}")
    return urlretrieve(url, filename)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version of the release being prepared, e.g. 3.3.0",
    )
    parser.add_argument(
        "--scala-version",
        type=str,
        required=True,
        help="Version of Scala to use in the JVM packages",
        choices=["2.12", "2.13"],
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["cpu", "gpu"],
        help="JVM package variant to package and publish",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use `mvn install` instead of `mvn deploy`.",
    )

    args = parser.parse_args()
    return args


def deploy_cuda_pkg(local: bool, version: str) -> None:
    """Deploy CUDA JVM packages."""
    url_prefix = "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/release/ml/dmlc"
    with tempfile.TemporaryDirectory() as tempdir:
        # libxgboost4j.so for Linux x86_64, GPU support
        zip_path = os.path.join(tempdir, "xgboost4j-spark-gpu_2.12.jar")
        extract_dir = os.path.join(tempdir, "xgboost4j-spark-gpu")
        retrieve(
            url=f"{url_prefix}/xgboost4j-spark-gpu_2.12/{version}/"
            f"xgboost4j-spark-gpu_2.12-{version}.jar",
            filename=zip_path,
        )
        os.mkdir(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as t:
            t.extractall(extract_dir)
        cp(
            os.path.join(extract_dir, "lib", "linux", "x86_64", "libxgboost4j.so"),
            "xgboost4j/src/main/resources/lib/linux/x86_64/libxgboost4j.so",
        )
    run(
        "mvn --no-transfer-progress install -Pgpu "
        "-DskipTests -Dmaven.test.skip=true -Dskip.native.build=true"
    )
    deploy(local, "gpu", "xgboost4j-spark-gpu")


def main() -> None:
    """Assemble and deploy the packages."""
    args = parse_args()

    scala_version = args.scala_version
    use_cuda = args.variant == "gpu"

    commit_hash = get_current_commit_hash()
    git_branch = get_current_git_branch()
    print(f"Using commit {commit_hash} of branch {git_branch}")
    print(f"====Update pom.xml to use Scala {scala_version}====")
    run(
        f"{sys.executable} ops/script/change_scala_version.py "
        f"--scala-version {scala_version} --purge-artifacts"
    )

    with chdir("jvm-packages/"):
        print("====Copying resources for testing====")
        with chdir("../demo/data/regression"):
            run(f"{sys.executable} mapfeat.py")
            run(f"{sys.executable} mknfold.py machine.txt 1")
        xgboost4j_spark = "xgboost4j-spark-gpu" if use_cuda else "xgboost4j-spark"
        maybe_makedirs("xgboost4j/src/test/resources")
        maybe_makedirs(f"{xgboost4j_spark}/src/test/resources")
        for file in glob.glob("../demo/data/agaricus.*"):
            cp(file, "xgboost4j/src/test/resources")
            cp(file, f"{xgboost4j_spark}/src/test/resources")
        for file in glob.glob("../demo/data/regression/machine.txt.t*"):
            cp(file, f"{xgboost4j_spark}/src/test/resources")

        print("====Creating directories to hold native binaries====")
        if use_cuda:
            # TODO(hcho3): Add GPU build for linux aarch64
            matrix = [("linux", "x86_64")]
        else:
            matrix = [
                ("linux", "x86_64"),
                ("linux", "aarch64"),
                ("windows", "x86_64"),
                ("macos", "x86_64"),
                ("macos", "aarch64"),
            ]
        for os_ident, arch in matrix:
            output_dir = f"xgboost4j/src/main/resources/lib/{os_ident}/{arch}"
            maybe_makedirs(output_dir)

        print("====Downloading native binaries from CI====")
        if use_cuda:
            deploy_cuda_pkg(args.local, args.release)
        else:
            url_prefix = "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds"
            for os_ident, arch, src_libname, dest_libname in [
                ("linux", "x86_64", "libxgboost4j_linux_x86_64.so", "libxgboost4j.so"),
                (
                    "linux",
                    "aarch64",
                    "libxgboost4j_linux_aarch64.so",
                    "libxgboost4j.so",
                ),
                ("windows", "x86_64", "xgboost4j.dll", "xgboost4j.dll"),
                ("macos", "x86_64", "libxgboost4j_intel.dylib", "libxgboost4j.dylib"),
                ("macos", "aarch64", "libxgboost4j_m1.dylib", "libxgboost4j.dylib"),
            ]:
                retrieve(
                    url=f"{url_prefix}/{git_branch}/{commit_hash}/{src_libname}",
                    filename=(
                        "xgboost4j/src/main/resources/lib/"
                        f"{os_ident}/{arch}/{dest_libname}"
                    ),
                )
            deploy(args.local, "default", None)

    if args.local:
        return

    print("====Next Steps====")
    print(
        "Visit https://central.sonatype.com/publishing/deployments to verify the deployment. "
        "You can either drop the deployment or publish it. Note: publishing is final."
    )


if __name__ == "__main__":
    main()
