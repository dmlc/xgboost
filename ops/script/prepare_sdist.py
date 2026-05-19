"""Stage the C++ source tree under python-package/cpp_src/ for sdist building.

`scikit-build-core`'s `sdist.include` patterns are relative to the project root
(`python-package/`) and cannot reach files above it. This script copies the C++
sources, headers, CMake files, plugins, and LICENSE that `pip install
xgboost-x.y.z.tar.gz` needs into `python-package/cpp_src/`, so the resulting
sdist is self-contained.

Run before `python -m build --sdist`. Idempotent: pass `--clean` to wipe an
existing staging directory first.
"""

import argparse
import pathlib
import shutil

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STAGING_DIR = REPO_ROOT / "python-package" / "cpp_src"

# Mirrors the old packager/sdist.py:copy_cpp_src_tree.
CPP_SUBDIRS = ["src", "include", "dmlc-core", "cmake", "plugin"]
CPP_FILES = ["CMakeLists.txt", "LICENSE"]


def stage(clean: bool = False) -> None:
    if STAGING_DIR.exists():
        if not clean:
            raise SystemExit(
                f"{STAGING_DIR} already exists. Pass --clean to overwrite."
            )
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    for sub in CPP_SUBDIRS:
        src = REPO_ROOT / sub
        if not src.is_dir():
            raise SystemExit(f"Expected directory {src} to exist.")
        shutil.copytree(src, STAGING_DIR / sub)
        print(f"copy  {src} -> {STAGING_DIR / sub}")
    for f in CPP_FILES:
        src = REPO_ROOT / f
        if not src.is_file():
            raise SystemExit(f"Expected file {src} to exist.")
        shutil.copy(src, STAGING_DIR / f)
        print(f"copy  {src} -> {STAGING_DIR / f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove an existing cpp_src/ before staging.",
    )
    args = parser.parse_args()
    stage(clean=args.clean)
