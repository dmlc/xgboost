"""
Simple script for managing Python, R, and source release packages.

tqdm, sh, and build are required to run this script.
"""

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import tqdm
from packaging import version
from pypi_variants import make_pyproject
from sh.contrib import git
from test_utils import PY_PACKAGE
from test_utils import ROOT as root_path
from test_utils import DirectoryExcursion

# S3 bucket hosting the release artifacts
S3_BUCKET_URL = "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds"
DIST = Path(PY_PACKAGE) / "dist"
ROOT = Path(root_path)

pbar = None


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


def retrieve(url: str, filename: Optional[Path] = None) -> str:
    """Retrieve a file from a URL with progress indication."""
    print(f"Downloading {url} -> {filename}")
    return urlretrieve(url, filename, reporthook=show_progress)[0]


def latest_hash() -> str:
    """Get latest commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to get latest commit hash.") from e


def _download_python_wheels(
    platforms: List[str],
    dir_url: str,
    src_filename_prefix: str,
    target_filename_prefix: str,
    outdir: Path,
) -> List[Path]:
    """Download all Python binary wheels for a given set of platforms"""
    wheel_paths = []
    dist_dir = outdir / "dist"
    dist_dir.mkdir(exist_ok=True)

    for platform in platforms:
        src_wheel = f"{src_filename_prefix}{platform}.whl"
        url = f"{dir_url}{src_wheel}"
        target_wheel = f"{target_filename_prefix}{platform}.whl"
        wheel_path = dist_dir / target_wheel
        wheel_paths.append(wheel_path)

        retrieve(url=url, filename=wheel_path)

        try:
            result = subprocess.run(
                ["twine", "check", str(wheel_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if "warning" in result.stderr or "warning" in result.stdout:
                raise RuntimeError(
                    f"Unresolved warnings:\n{result.stderr}\n{result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed twine check") from e
    return wheel_paths


def make_python_sdist(
    release: str, rc: Optional[str], rc_ver: Optional[int], outdir: Path
) -> None:
    """Make Python source distribution."""
    dist_dir = outdir / "dist"
    dist_dir.mkdir(exist_ok=True)

    # Build sdist for `xgboost-cpu`.
    with DirectoryExcursion(ROOT):
        make_pyproject("cpu")
    with DirectoryExcursion(ROOT / "python-package"):
        subprocess.run(["python", "-m", "build", "--sdist"], check=True)
        sdist_name = (
            f"xgboost_cpu-{release}{rc}{rc_ver}.tar.gz"
            if rc
            else f"xgboost_cpu-{release}.tar.gz"
        )
        src = DIST / sdist_name
        subprocess.run(["twine", "check", str(src)], check=True)
        dest = dist_dir / sdist_name
        shutil.move(src, dest)

    # Build sdist for `xgboost`.
    with DirectoryExcursion(ROOT):
        make_pyproject("default")

    with DirectoryExcursion(ROOT / "python-package"):
        subprocess.run(["python", "-m", "build", "--sdist"], check=True)
        sdist_name = (
            f"xgboost-{release}{rc}{rc_ver}.tar.gz"
            if rc
            else f"xgboost-{release}.tar.gz"
        )
        src = DIST / sdist_name
        subprocess.run(["twine", "check", str(src)], check=True)
        dest = dist_dir / sdist_name
        shutil.move(src, dest)


def download_python_wheels(branch: str, commit_hash: str, outdir: Path) -> None:
    """Download all Python binary wheels for the specified branch."""
    full_platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "manylinux_2_28_x86_64",
        "manylinux_2_28_aarch64",
        "macosx_10_15_x86_64",
        "macosx_12_0_arm64",
    ]
    minimal_platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
    ]

    # https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_3.0.0/4bfd4bf60d32e2d62426cc4070ccb5a5ba1ed078/xgboost-3.0.0rc1-py3-none-manylinux_2_28_x86_64.whl
    dir_url = f"{S3_BUCKET_URL}/{branch}/{commit_hash}/"
    wheels = []
    for pkg_name, platforms in [
        ("xgboost", full_platforms),
        ("xgboost_cpu", minimal_platforms),
    ]:
        src_filename_prefix = f"{pkg_name}-{args.release}-py3-none-"
        target_filename_prefix = f"{pkg_name}-{args.release}-py3-none-"
        wheels.extend(
            _download_python_wheels(
                platforms, dir_url, src_filename_prefix, target_filename_prefix, outdir
            )
        )
    print(f"List of downloaded wheels: {wheels}")
    print(
        """
Following steps should be done manually:
- Upload pypi package by `python3 -m twine upload dist/<Package Name>` for all wheels.
- Check the uploaded files on `https://pypi.org/project/xgboost/<VERSION>/#files` and
  `pip install xgboost==<VERSION>` """
    )


def download_r_artifacts(
    release: str, branch: str, commit: str, outdir: Path
) -> Tuple[Dict[str, str], List[str]]:
    """Download R package artifacts for the specified release and branch."""
    platforms = ["linux"]
    rpkg_dir = outdir / "r-packages"
    rpkg_dir.mkdir(exist_ok=True)

    artifacts = []
    urls = {}

    # https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_3.0.0/4bfd4bf60d32e2d62426cc4070ccb5a5ba1ed078/xgboost_r_gpu_linux.tar.gz
    for plat in platforms:
        url = f"{S3_BUCKET_URL}/{branch}/{commit}/xgboost_r_gpu_{plat}.tar.gz"
        artifact_name = f"xgboost_r_gpu_{plat}.tar.gz"
        artifact_path = rpkg_dir / artifact_name
        retrieve(url=url, filename=artifact_path)
        artifacts.append(artifact_path)
        urls[plat] = url

    print(f"Finished downloading R package artifacts: {artifacts}")
    hashes = []
    with DirectoryExcursion(rpkg_dir):
        for f in artifacts:
            result = subprocess.run(
                ["sha256sum", f.name],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            hashes.append(result.stdout.strip())
    return urls, hashes


def check_path() -> None:
    """Ensure the script is run from the project root directory."""
    current_dir = Path.cwd().resolve()
    if current_dir.name != "xgboost":
        raise RuntimeError("Must be run from the project root directory.")


def make_src_tarball(release: str, outdir: Path) -> Tuple[str, str]:
    tarball_name = f"xgboost-src-{release}.tar.gz"
    tarball_path = outdir / tarball_name
    if tarball_path.exists():
        tarball_path.unlink()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shutil.copytree(Path.cwd(), tmpdir / "xgboost")
        with DirectoryExcursion(tmpdir / "xgboost"):
            result = subprocess.run(
                ["git", "submodule", "foreach", "--quiet", "echo $sm_path"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            submodules = result.stdout.strip().split()
            for mod in submodules:
                mod_path = Path.cwd().resolve() / mod / ".git"
                mod_path.unlink()
            shutil.rmtree(".git")
            with tarfile.open(tarball_path, "x:gz") as tar:
                tar.add(tmpdir / "xgboost", arcname="xgboost")

    with DirectoryExcursion(tarball_path.parent):
        result = subprocess.run(
            ["sha256sum", tarball_name],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        sha256sum = result.stdout.strip()
    return tarball_name, sha256sum


def release_note(
    release: str,
    artifact_hashes: List[str],
    r_urls: Dict[str, str],
    tarball_name: str,
    outdir: Path,
) -> None:
    """Generate a note for GitHub release description."""
    r_gpu_linux_url = r_urls["linux"]
    src_tarball = (
        f"https://github.com/dmlc/xgboost/releases/download/v{release}/{tarball_name}"
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
* xgboost_r_gpu_linux_{release}.tar.gz: [Download]({r_gpu_linux_url})

**Source tarball**
* {tarball_name}: [Download]({src_tarball})"""
    print(end_note)
    with open(outdir / "end_note.md", "w") as f:
        f.write(end_note)


def main(args: argparse.Namespace) -> None:
    check_path()

    release_parsed: version.Version = version.parse(args.release)
    print(f"Release: {release_parsed}")

    major = release_parsed.major
    minor = release_parsed.minor
    patch = release_parsed.micro
    if not release_parsed.is_prerelease:
        # Major release
        rc: Optional[str] = None
        rc_ver: Optional[int] = None
    else:
        # RC release
        assert release_parsed.pre is not None
        rc, rc_ver = release_parsed.pre
        if rc != "rc":
            raise ValueError(
                "Only supports release candidates with 'rc' in the version string"
            )

    # Release string with only major, minor, patch components
    release = f"{major}.{minor}.{patch}"
    if args.branch is not None:
        branch = args.branch
    else:
        branch = f"release_{major}.{minor}.0"

    git.clean("-xdf")
    git.checkout(branch)
    git.pull("origin", branch)
    git.submodule("update")
    commit_hash = latest_hash()

    outdir = Path(args.outdir).resolve()
    if ROOT in outdir.parents:
        raise ValueError("Output directory must be outside of the source tree.")
    outdir.mkdir(exist_ok=True)

    artifact_hashes: List[str] = []

    # Source tarball
    tarball_name, hash = make_src_tarball(release, outdir)
    artifact_hashes.append(hash)

    # CUDA R packages
    urls, hashes = download_r_artifacts(
        release,
        branch,
        commit_hash,
        outdir,
    )
    artifact_hashes.extend(hashes)

    # Python source wheel
    make_python_sdist(release, rc, rc_ver, outdir)

    # Python binary wheels
    download_python_wheels(branch, commit_hash, outdir)

    # Write end note
    release_note(release, artifact_hashes, urls, tarball_name, outdir)


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
