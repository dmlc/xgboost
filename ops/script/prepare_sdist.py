"""Stage the C++ source tree under python-package/cpp_src/ for sdist building.

`scikit-build-core`'s `sdist.include` patterns are relative to the project root
(`python-package/`) and cannot reach files above it. This script copies the C++
sources, headers, CMake files, plugins, and LICENSE that `pip install
xgboost-x.y.z.tar.gz` needs into `python-package/cpp_src/`, so the resulting
sdist is self-contained.

Run before `python -m build --sdist`. Idempotent: this script wipes the
existing staging directory first.
"""

import pathlib
import shutil

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STAGING_DIR = REPO_ROOT / "python-package" / "cpp_src"

CPP_SUBDIRS = ["src", "include", "dmlc-core", "cmake", "plugin"]
CPP_FILES = ["CMakeLists.txt", "LICENSE"]


def stage() -> None:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    for sub in CPP_SUBDIRS:
        src = REPO_ROOT / sub
        if not src.is_dir():
            raise SystemExit(f"Expected directory '{src}' to exist.")
        shutil.copytree(src, STAGING_DIR / sub)
        print(f"Copy {src} -> {STAGING_DIR / sub}")
    for f in CPP_FILES:
        src = REPO_ROOT / f
        shutil.copy(src, STAGING_DIR / f)
        print(f"Copy {src} -> {STAGING_DIR / f}")


if __name__ == "__main__":
    stage()
