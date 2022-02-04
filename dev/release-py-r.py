"""Simple script for downloading and checking pypi release wheels.

tqdm, sh are required to run this script.
"""
from urllib.request import urlretrieve
from typing import cast, Tuple
import argparse
from typing import List
from sh.contrib import git
from distutils import version
import subprocess
import tqdm
import os

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
        pbar.update(block_size / 1024)
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


def download_py_packages(major: int, minor: int, commit_hash: str):
    platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64"
    ]

    dir_URL = PREFIX + str(major) + "." + str(minor) + ".0" + "/"
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


def download_r_packages(release: str, rc: str, commit: str) -> None:
    platforms = ["win64", "linux"]
    dirname = "./r-packages"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filenames = []

    for plat in platforms:
        url = f"{PREFIX}{release}/xgboost_r_gpu_{plat}_{commit}.tar.gz"

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

    rel = version.LooseVersion(args.release)

    print("Release:", rel)
    if len(rel.version) == 3:
        # Major release
        major, minor, patch = version.StrictVersion(args.release).version
        rc = None
        rc_ver = None
    else:
        # RC release
        major, minor, patch, rc, rc_ver = cast(
            Tuple[int, int, int, str, int], rel.version
        )
        assert rc == "rc"

    release = str(major) + "." + str(minor) + "." + str(patch)
    branch = "release_" + release
    git.clean("-xdf")
    git.checkout(branch)
    git.pull("origin", branch)
    git.submodule("update")
    commit_hash = latest_hash()

    download_r_packages(
        release, "" if rc is None else rc + str(rc_ver), commit_hash
    )

    download_py_packages(major, minor, commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version tag, e.g. '1.3.2', or '1.5.0rc1'"
    )
    args = parser.parse_args()
    main(args)
