"""Simple script for downloading and checking pypi release wheels.

tqdm, sh are required to run this script.
"""
import argparse
import os
import subprocess
from typing import List, Optional
from urllib.request import urlretrieve

import tqdm
from packaging import version
from sh.contrib import git

# The package building is managed by Jenkins CI.
PREFIX = "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_"
DIST = os.path.join(os.path.curdir, "python-package", "dist")

pbar = None


def show_progress(block_num, block_size, total_size):
    "Show file download progress."
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

    downloaded = block_num * block_size
    if downloaded < total_size:
        upper = (total_size - downloaded) / 1024
        pbar.update(min(block_size / 1024, upper))
    else:
        pbar.close()
        pbar = None


def retrieve(url, filename=None):
    print(f"{url} -> {filename}")
    return urlretrieve(url, filename, reporthook=show_progress)


def latest_hash() -> str:
    "Get latest commit hash."
    ret = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
    assert ret.returncode == 0, "Failed to get latest commit hash."
    commit_hash = ret.stdout.decode("utf-8").strip()
    return commit_hash


def download_wheels(
    platforms: List[str],
    dir_URL: str,
    src_filename_prefix: str,
    target_filename_prefix: str,
) -> List[str]:
    """Download all binary wheels. dir_URL is the URL for remote directory storing the release
    wheels

    """

    filenames = []
    for platform in platforms:
        src_wheel = src_filename_prefix + platform + ".whl"
        url = dir_URL + src_wheel

        target_wheel = target_filename_prefix + platform + ".whl"
        filename = os.path.join(DIST, target_wheel)
        filenames.append(filename)
        retrieve(url=url, filename=filename)
        ret = subprocess.run(["twine", "check", filename], capture_output=True)
        assert ret.returncode == 0, "Failed twine check"
        stderr = ret.stderr.decode("utf-8")
        stdout = ret.stdout.decode("utf-8")
        assert stderr.find("warning") == -1, "Unresolved warnings:\n" + stderr
        assert stdout.find("warning") == -1, "Unresolved warnings:\n" + stdout
    return filenames


def download_py_packages(branch: str, major: int, minor: int, commit_hash: str) -> None:
    platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64"
    ]

    branch = branch.split("_")[1]  # release_x.y.z
    dir_URL = PREFIX + branch + "/"
    src_filename_prefix = "xgboost-" + args.release + "%2B" + commit_hash + "-py3-none-"
    target_filename_prefix = "xgboost-" + args.release + "-py3-none-"

    if not os.path.exists(DIST):
        os.mkdir(DIST)

    filenames = download_wheels(
        platforms, dir_URL, src_filename_prefix, target_filename_prefix
    )
    print("List of downloaded wheels:", filenames)
    print(
        """
Following steps should be done manually:
- Generate source package by running `python setup.py sdist`.
- Upload pypi package by `python3 -m twine upload dist/<Package Name>` for all wheels.
- Check the uploaded files on `https://pypi.org/project/xgboost/<VERSION>/#files` and `pip
  install xgboost==<VERSION>` """
    )


def download_r_packages(release: str, branch: str, rc: str, commit: str) -> None:
    platforms = ["win64", "linux"]
    dirname = "./r-packages"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filenames = []
    branch = branch.split("_")[1]  # release_x.y.z

    for plat in platforms:
        url = f"{PREFIX}{branch}/xgboost_r_gpu_{plat}_{commit}.tar.gz"

        if not rc:
            filename = f"xgboost_r_gpu_{plat}_{release}.tar.gz"
        else:
            filename = f"xgboost_r_gpu_{plat}_{release}-{rc}.tar.gz"

        target = os.path.join(dirname, filename)
        retrieve(url=url, filename=target)
        filenames.append(target)

    print("Finished downloading R packages:", filenames)


def check_path():
    root = os.path.abspath(os.path.curdir)
    assert os.path.basename(root) == "xgboost", "Must be run on project root."


def main(args: argparse.Namespace) -> None:
    check_path()

    rel = version.parse(args.release)
    assert isinstance(rel, version.Version)

    major = rel.major
    minor = rel.minor
    patch = rel.micro

    print("Release:", rel)
    if not rel.is_prerelease:
        # Major release
        rc: Optional[str] = None
        rc_ver: Optional[int] = None
    else:
        # RC release
        major = rel.major
        minor = rel.minor
        patch = rel.micro
        assert rel.pre is not None
        rc, rc_ver = rel.pre
        assert rc == "rc"

    release = str(major) + "." + str(minor) + "." + str(patch)
    if args.branch is not None:
        branch = args.branch
    else:
        branch = "release_" + str(major) + "." + str(minor) + ".0"

    git.clean("-xdf")
    git.checkout(branch)
    git.pull("origin", branch)
    git.submodule("update")
    commit_hash = latest_hash()

    download_r_packages(
        release, branch, "" if rc is None else rc + str(rc_ver), commit_hash
    )

    download_py_packages(branch, major, minor, commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version tag, e.g. '1.3.2', or '1.5.0rc1'"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help=(
            "Optional branch. Usually patch releases reuse the same branch of the"
            " major release, but there can be exception."
        )
    )
    args = parser.parse_args()
    main(args)
