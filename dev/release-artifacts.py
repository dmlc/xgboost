"""Simple script for managing Python, R, and source release packages.

tqdm, sh are required to run this script.
"""
import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import tqdm
from packaging import version
from sh.contrib import git

# The package building is managed by Jenkins CI.
PREFIX = "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_"
ROOT = Path(__file__).absolute().parent.parent
DIST = ROOT / "python-package" / "dist"

pbar = None


class DirectoryExcursion:
    def __init__(self, path: Union[os.PathLike, str]) -> None:
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self) -> None:
        os.chdir(self.path)

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)


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
    outdir: str,
) -> List[str]:
    """Download all binary wheels. dir_URL is the URL for remote directory storing the
    release wheels.

    """

    filenames = []
    outdir = os.path.join(outdir, "dist")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for platform in platforms:
        src_wheel = src_filename_prefix + platform + ".whl"
        url = dir_URL + src_wheel

        target_wheel = target_filename_prefix + platform + ".whl"
        filename = os.path.join(outdir, target_wheel)
        filenames.append(filename)
        retrieve(url=url, filename=filename)
        ret = subprocess.run(["twine", "check", filename], capture_output=True)
        assert ret.returncode == 0, "Failed twine check"
        stderr = ret.stderr.decode("utf-8")
        stdout = ret.stdout.decode("utf-8")
        assert stderr.find("warning") == -1, "Unresolved warnings:\n" + stderr
        assert stdout.find("warning") == -1, "Unresolved warnings:\n" + stdout
    return filenames


def make_pysrc_wheel(release: str, outdir: str) -> None:
    """Make Python source distribution."""
    dist = os.path.join(outdir, "dist")
    if not os.path.exists(dist):
        os.mkdir(dist)

    with DirectoryExcursion(os.path.join(ROOT, "python-package")):
        subprocess.check_call(["python", "-m", "build", "--sdist"])
        src = os.path.join(DIST, f"xgboost-{release}.tar.gz")
        subprocess.check_call(["twine", "check", src])
        shutil.move(src, os.path.join(dist, f"xgboost-{release}.tar.gz"))


def download_py_packages(
    branch: str, major: int, minor: int, commit_hash: str, outdir: str
) -> None:
    platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64",
    ]

    branch = branch.split("_")[1]  # release_x.y.z
    dir_URL = PREFIX + branch + "/"
    src_filename_prefix = "xgboost-" + args.release + "%2B" + commit_hash + "-py3-none-"
    target_filename_prefix = "xgboost-" + args.release + "-py3-none-"

    if not os.path.exists(DIST):
        os.mkdir(DIST)

    filenames = download_wheels(
        platforms, dir_URL, src_filename_prefix, target_filename_prefix, outdir
    )
    print("List of downloaded wheels:", filenames)
    print(
        """
Following steps should be done manually:
- Upload pypi package by `python3 -m twine upload dist/<Package Name>` for all wheels.
- Check the uploaded files on `https://pypi.org/project/xgboost/<VERSION>/#files` and
  `pip install xgboost==<VERSION>` """
    )


def download_r_packages(
    release: str, branch: str, rc: str, commit: str, outdir: str
) -> Tuple[Dict[str, str], List[str]]:
    platforms = ["win64", "linux"]
    dirname = os.path.join(outdir, "r-packages")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filenames = []
    branch = branch.split("_")[1]  # release_x.y.z
    urls = {}

    for plat in platforms:
        url = f"{PREFIX}{branch}/xgboost_r_gpu_{plat}_{commit}.tar.gz"

        if not rc:
            filename = f"xgboost_r_gpu_{plat}_{release}.tar.gz"
        else:
            filename = f"xgboost_r_gpu_{plat}_{release}-{rc}.tar.gz"

        target = os.path.join(dirname, filename)
        retrieve(url=url, filename=target)
        filenames.append(target)
        urls[plat] = url

    print("Finished downloading R packages:", filenames)
    hashes = []
    with DirectoryExcursion(os.path.join(outdir, "r-packages")):
        for f in filenames:
            ret = subprocess.run(["sha256sum", os.path.basename(f)], capture_output=True)
            h = ret.stdout.decode().strip()
            hashes.append(h)
    return urls, hashes


def check_path():
    root = os.path.abspath(os.path.curdir)
    assert os.path.basename(root) == "xgboost", "Must be run on project root."


def make_src_package(release: str, outdir: str) -> Tuple[str, str]:
    tarname = f"xgboost-{release}.tar.gz"
    tarpath = os.path.join(outdir, tarname)
    if os.path.exists(tarpath):
        os.remove(tarpath)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shutil.copytree(os.path.curdir, tmpdir / "xgboost")
        with DirectoryExcursion(tmpdir / "xgboost"):
            ret = subprocess.run(
                ["git", "submodule", "foreach", "--quiet", "echo $sm_path"],
                capture_output=True,
            )
            submodules = ret.stdout.decode().strip().split()
            for mod in submodules:
                mod_path = os.path.join(os.path.abspath(os.path.curdir), mod, ".git")
                os.remove(mod_path)
            shutil.rmtree(".git")
            with tarfile.open(tarpath, "x:gz") as tar:
                src = tmpdir / "xgboost"
                tar.add(src, arcname="xgboost")

    with DirectoryExcursion(os.path.dirname(tarpath)):
        ret = subprocess.run(["sha256sum", tarname], capture_output=True)
        h = ret.stdout.decode().strip()
    return tarname, h


def release_note(
    release: str,
    artifact_hashes: List[str],
    r_urls: Dict[str, str],
    tarname: str,
    outdir: str,
) -> None:
    """Generate a note for GitHub release description."""
    r_gpu_linux_url = r_urls["linux"]
    r_gpu_win64_url = r_urls["win64"]
    src_tarball = (
        f"https://github.com/dmlc/xgboost/releases/download/v{release}/{tarname}"
    )
    hash_note = "\n".join(artifact_hashes)

    end_note = f"""
### Additional artifacts:

You can verify the downloaded packages by running the following command on your Unix shell:

``` sh
echo "<hash> <artifact>" | shasum -a 256 --check
```

```
{hash_note}
```

**Experimental binary packages for R with CUDA enabled**
* xgboost_r_gpu_linux_1.7.5.tar.gz: [Download]({r_gpu_linux_url})
* xgboost_r_gpu_win64_1.7.5.tar.gz: [Download]({r_gpu_win64_url})

**Source tarball**
* xgboost.tar.gz: [Download]({src_tarball})"""
    print(end_note)
    with open(os.path.join(outdir, "end_note.md"), "w") as fd:
        fd.write(end_note)


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

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # source tarball
    hashes: List[str] = []
    tarname, h = make_src_package(release, args.outdir)
    hashes.append(h)

    # CUDA R packages
    urls, hr = download_r_packages(
        release,
        branch,
        "" if rc is None else rc + str(rc_ver),
        commit_hash,
        args.outdir,
    )
    hashes.extend(hr)

    # Python source wheel
    make_pysrc_wheel(release, args.outdir)

    # Python binary wheels
    download_py_packages(branch, major, minor, commit_hash, args.outdir)

    # Write end note
    release_note(release, hashes, urls, tarname, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version tag, e.g. '1.3.2', or '1.5.0rc1'",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help=(
            "Optional branch. Usually patch releases reuse the same branch of the"
            " major release, but there can be exception."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        required=True,
        help="Directory to store the generated packages.",
    )
    args = parser.parse_args()
    main(args)
