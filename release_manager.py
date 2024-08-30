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
        self.curdir = os.path.abspath(os.getcwd())

    def __enter__(self) -> None:
        os.chdir(self.path)

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Show file download progress."""
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(min(block_size / 1024, (total_size - downloaded) / 1024))
    else:
        pbar.close()
        pbar = None


def retrieve(url: str, filename: Optional[str] = None) -> str:
    """Retrieve a file from a URL with progress indication."""
    print(f"Downloading {url} -> {filename}")
    return urlretrieve(url, filename, reporthook=show_progress)[0]


def latest_hash() -> str:
    """Get the latest commit hash."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to get latest commit hash.") from e


def download_wheels(
    platforms: List[str],
    dir_url: str,
    src_filename_prefix: str,
    target_filename_prefix: str,
    outdir: str
) -> List[str]:
    """Download all binary wheels from the remote directory."""
    filenames = []
    dist_dir = os.path.join(outdir, "dist")
    os.makedirs(dist_dir, exist_ok=True)

    for platform in platforms:
        src_wheel = f"{src_filename_prefix}{platform}.whl"
        url = f"{dir_url}{src_wheel}"
        target_wheel = f"{target_filename_prefix}{platform}.whl"
        filename = os.path.join(dist_dir, target_wheel)
        filenames.append(filename)

        retrieve(url=url, filename=filename)

        try:
            result = subprocess.run(["twine", "check", filename], capture_output=True, text=True, check=True)
            stderr = result.stderr
            stdout = result.stdout
            if "warning" in stderr or "warning" in stdout:
                raise RuntimeError(f"Unresolved warnings:\n{stderr}\n{stdout}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed twine check") from e

    return filenames


def make_pysrc_wheel(
    release: str, rc: Optional[str], rc_ver: Optional[int], outdir: str
) -> None:
    """Make Python source distribution and handle NCCL dependency removal."""
    dist_dir = os.path.join(outdir, "dist")
    os.makedirs(dist_dir, exist_ok=True)

    with DirectoryExcursion(ROOT):
        pyproject_path = "python-package/pyproject.toml"
        with open(pyproject_path, "r") as f:
            orig_pyproj_lines = f.read()
        with open("tests/buildkite/remove_nccl_dep.patch", "r") as f:
            patch_lines = f.read()
        subprocess.run(["patch", "-p0"], input=patch_lines, text=True)

    with DirectoryExcursion(ROOT / "python-package"):
        subprocess.check_call(["python", "-m", "build", "--sdist"])
        name = f"xgboost-{release}{rc}{rc_ver}.tar.gz" if rc else f"xgboost-{release}.tar.gz"
        src = os.path.join(DIST, name)
        subprocess.check_call(["twine", "check", src])
        target = os.path.join(dist_dir, name)
        shutil.move(src, target)

    with DirectoryExcursion(ROOT):
        with open("python-package/pyproject.toml", "w") as f:
            f.write(orig_pyproj_lines)


def download_py_packages(
    branch: str, commit_hash: str, outdir: str
) -> None:
    """Download Python packages for the specified branch."""
    full_platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "manylinux_2_28_x86_64",
        "manylinux_2_28_aarch64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64",
    ]
    minimal_platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
    ]

    branch = branch.split("_")[1]
    dir_url = f"{PREFIX}{branch}/"
    wheels = []

    for pkg_name, platforms in [("xgboost", full_platforms), ("xgboost_cpu", minimal_platforms)]:
        src_filename_prefix = f"{pkg_name}-{args.release}%2B{commit_hash}-py3-none-"
        target_filename_prefix = f"{pkg_name}-{args.release}-py3-none-"
        wheels.extend(download_wheels(
            platforms, dir_url, src_filename_prefix, target_filename_prefix, outdir
        ))
    
    print("List of downloaded wheels:", wheels)
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
    """Download R packages for the specified release and branch."""
    platforms = ["linux"]
    r_package_dir = os.path.join(outdir, "r-packages")
    os.makedirs(r_package_dir, exist_ok=True)

    filenames = []
    urls = {}

    for plat in platforms:
        url = f"{PREFIX}{branch}/xgboost_r_gpu_{plat}_{commit}.tar.gz"
        filename = f"xgboost_r_gpu_{plat}_{release}-{rc}.tar.gz" if rc else f"xgboost_r_gpu_{plat}_{release}.tar.gz"
        target = os.path.join(r_package_dir, filename)
        retrieve(url=url, filename=target)
        filenames.append(target)
        urls[plat] = url

    print("Finished downloading R packages:", filenames)
    hashes = []
    with DirectoryExcursion(r_package_dir):
        for f in filenames:
            result = subprocess.run(["sha256sum", os.path.basename(f)], capture_output=True, text=True, check=True)
            hashes.append(result.stdout.strip())
    return urls, hashes


def check_path() -> None:
    """Ensure the script is run from the project root directory."""
    root = os.path.abspath(os.getcwd())
    if os.path.basename(root) != "xgboost":
        raise RuntimeError("Must be run from the project root directory.")


def make_src_package(release: str, outdir: str) -> Tuple[str, str]:
    """Create a source package tarball."""
    tarname = f"xgboost-{release}.tar.gz"
    tarpath = os.path.join(outdir, tarname)
    if os.path.exists(tarpath):
        os.remove(tarpath)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shutil.copytree(os.getcwd(), tmpdir / "xgboost")
        with DirectoryExcursion(tmpdir / "xgboost"):
            result = subprocess.run(
                ["git", "submodule", "foreach", "--quiet", "echo $sm_path"],
                capture_output=True, text=True, check=True
            )
            submodules = result.stdout.strip().split()
            for mod in submodules:
                mod_path = os.path.join(os.path.abspath(os.getcwd()), mod, ".git")
                os.remove(mod_path)
            shutil.rmtree(".git")
            with tarfile.open(tarpath, "x:gz") as tar:
                tar.add(tmpdir / "xgboost", arcname="xgboost")

    with DirectoryExcursion(os.path.dirname(tarpath)):
        result = subprocess.run(["twine", "check", tarname], capture_output=True, text=True, check=True)
        stderr = result.stderr
        stdout = result.stdout
        if "warning" in stderr or "warning" in stdout:
            raise RuntimeError(f"Unresolved warnings:\n{stderr}\n{stdout}")

    return tarpath, tarname


def main() -> None:
    """Main function to handle command-line arguments and run the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="Release version")
    parser.add_argument("--branch", default="master", help="Branch name")
    parser.add_argument("--rc", default="", help="Release candidate (rc) version")
    parser.add_argument("--rc-ver", type=int, default=0, help="Release candidate version number")
    parser.add_argument("--commit-hash", default=latest_hash(), help="Commit hash")
    args = parser.parse_args()

    check_path()
    outdir = os.path.join(ROOT, "dist")
    os.makedirs(outdir, exist_ok=True)

    make_pysrc_wheel(
        args.release, args.rc, args.rc_ver, outdir
    )
    download_py_packages(
        args.branch, args.commit_hash, outdir
    )
    urls, hashes = download_r_packages(
        args.release, args.branch, args.rc, args.commit_hash, outdir
    )
    print(f"R package URLs:\n{urls}")
    print(f"R package hashes:\n{hashes}")
    src_tarpath, src_tarname = make_src_package(args.release, outdir)
    print(f"Source package created at {src_tarpath}")


if __name__ == "__main__":
    main()
