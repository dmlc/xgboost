#!/usr/bin/env python3
"""
Build XGBoost documentation locally or in CI.

This script provides a unified interface for building XGBoost documentation,
used by both local development and CI pipelines.

Components:
  - jvm-lib: Build libxgboost4j.so native library
  - jvm: Build JVM documentation (Scaladoc/Javadoc)
  - r: Build R documentation (pkgdown)
  - sphinx: Build Sphinx documentation (Python/C++)

Examples:
  # Local development - build everything
  python build_local_docs.py

  # Local development - Python docs only (fastest)
  python build_local_docs.py --skip-r --skip-jvm --skip-cpp

  # Local development - with artifact reuse
  python build_local_docs.py --reuse

  # CI - build specific components
  python build_local_docs.py jvm-lib --cuda --sccache
  python build_local_docs.py jvm --branch-name PR-123
  python build_local_docs.py r --branch-name master
  python build_local_docs.py sphinx --r-docs r.tar.bz2 --jvm-docs jvm.tar.bz2
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


# =============================================================================
# Utilities
# =============================================================================


def get_project_root() -> Path:
    """Get the XGBoost project root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent.parent


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    return shutil.which(cmd) is not None


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print(f"\n>>> {' '.join(cmd)}")
    if cwd:
        print(f"    (in {cwd})")
    return subprocess.run(cmd, cwd=cwd, env=merged_env, check=check)


def check_dependencies(components: set[str]) -> bool:
    """Check for required dependencies."""
    missing = []

    if "cpp" in components:
        for cmd, pkg in [
            ("doxygen", "doxygen"),
            ("dot", "graphviz"),
            ("cmake", "cmake"),
            ("ninja", "ninja-build"),
        ]:
            if not check_command(cmd):
                missing.append(f"{cmd} (apt: {pkg})")
        if not check_command("g++") and not check_command("clang++"):
            missing.append("g++ or clang++")

    if "jvm-lib" in components or "jvm" in components:
        if not check_command("cmake"):
            missing.append("cmake")
        if not check_command("ninja"):
            missing.append("ninja (apt: ninja-build)")

    if "jvm" in components:
        if not check_command("mvn"):
            missing.append("mvn (apt: maven)")
        if not check_command("java"):
            missing.append("java (apt: openjdk-11-jdk)")

    if "r" in components:
        if not check_command("R"):
            missing.append("R (apt: r-base)")

    if missing:
        print("ERROR: Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        return False
    return True


# =============================================================================
# Build Functions
# =============================================================================


def build_jvm_lib(
    project_root: Path,
    cuda: bool = False,
    sccache: bool = False,
    gpu_arch: str | None = None,
) -> Path | None:
    """Build libxgboost4j.so native library."""
    print("\n" + "=" * 60)
    print("Building libxgboost4j.so")
    print("=" * 60)

    build_dir = project_root / ("build-gpu" if cuda else "build")
    build_dir.mkdir(exist_ok=True)

    cmake_args = ["cmake", "..", "-GNinja", "-DJVM_BINDINGS=ON"]

    if cuda:
        cmake_args.extend(["-DUSE_CUDA=ON", "-DUSE_NCCL=ON"])
        if gpu_arch:
            cmake_args.append(f"-DGPU_COMPUTE_VER={gpu_arch}")
    else:
        cmake_args.append("-DUSE_OPENMP=ON")

    if sccache:
        cmake_args.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ])
        if cuda:
            cmake_args.append("-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache")

    run_cmd(cmake_args, cwd=build_dir)
    run_cmd(["ninja", "xgboost4j"], cwd=build_dir)

    lib_path = project_root / "lib" / "libxgboost4j.so"
    if lib_path.exists():
        print(f"Built: {lib_path}")
        return lib_path
    print(f"ERROR: {lib_path} not found")
    return None


def build_jvm_docs(
    project_root: Path,
    branch_name: str = "local",
    create_tarball: bool = True,
) -> Path | None:
    """Build JVM documentation using Maven."""
    print("\n" + "=" * 60)
    print("Building JVM Documentation")
    print("=" * 60)

    lib_path = project_root / "lib" / "libxgboost4j.so"
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'jvm-lib' first.")
        return None

    jvm_dir = project_root / "jvm-packages"

    # Copy library to JVM resources
    res_dir = jvm_dir / "xgboost4j/src/main/resources/lib/linux/x86_64"
    res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(lib_path, res_dir / "libxgboost4j.so")

    # Build with Maven
    run_cmd(["mvn", "--no-transfer-progress", "install", "-Pdocs", "-DskipTests"], cwd=jvm_dir)
    run_cmd(["mvn", "--no-transfer-progress", "scala:doc", "-Pdocs"], cwd=jvm_dir)
    run_cmd(["mvn", "--no-transfer-progress", "javadoc:javadoc", "-Pdocs"], cwd=jvm_dir)

    if not create_tarball:
        return None

    # Package docs
    tmp_dir = jvm_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    (tmp_dir / "scaladocs").mkdir()

    apidocs = jvm_dir / "xgboost4j/target/reports/apidocs"
    if apidocs.exists():
        shutil.copytree(apidocs, tmp_dir / "javadocs")

    for pkg in ["xgboost4j", "xgboost4j-spark", "xgboost4j-spark-gpu", "xgboost4j-flink"]:
        src = jvm_dir / pkg / "target/site/scaladocs"
        if src.exists():
            shutil.copytree(src, tmp_dir / "scaladocs" / pkg)

    tarball = jvm_dir / f"{branch_name}.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(tmp_dir / "javadocs", arcname="javadocs")
        tar.add(tmp_dir / "scaladocs", arcname="scaladocs")
    shutil.rmtree(tmp_dir)

    print(f"Created: {tarball}")
    return tarball


def build_r_docs(
    project_root: Path,
    branch_name: str = "local",
    create_tarball: bool = True,
    r_libs_user: str | None = None,
) -> Path | None:
    """Build R documentation using pkgdown."""
    print("\n" + "=" * 60)
    print("Building R Documentation")
    print("=" * 60)

    r_pkg_dir = project_root / "R-package"
    r_doc_dir = project_root / "doc" / "R-package"

    # Setup R environment
    if r_libs_user:
        r_libs = Path(r_libs_user)
    else:
        r_libs = project_root / "r_libs_tmp"
    r_libs.mkdir(parents=True, exist_ok=True)

    env = {"R_LIBS_USER": str(r_libs), "MAKEFLAGS": f"-j{os.cpu_count()}"}

    # Build R docs
    run_cmd(["Rscript", "./tests/helper_scripts/install_deps.R"], cwd=r_pkg_dir, env=env)
    run_cmd(["Rscript", "-e", "pkgdown::build_site(examples=FALSE)"], cwd=r_pkg_dir, env=env)
    run_cmd(["R", "CMD", "INSTALL", "."], cwd=r_pkg_dir, env=env)
    run_cmd(["make", f"-j{os.cpu_count()}", "all"], cwd=r_doc_dir, env=env)

    if not create_tarball:
        return None

    # Package docs
    tarball = project_root / f"r-docs-{branch_name}.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(r_pkg_dir / "docs", arcname="R-package/docs")
        for md in ["xgboost_introduction.md", "xgboostfromJSON.md"]:
            md_path = r_doc_dir / md
            if md_path.exists():
                tar.add(md_path, arcname=f"doc/R-package/{md}")

    print(f"Created: {tarball}")
    return tarball


def build_sphinx_docs(
    project_root: Path,
    r_docs: Path | None = None,
    jvm_docs: Path | None = None,
    skip_r: bool = False,
    skip_jvm: bool = False,
    skip_cpp: bool = False,
    skip_deps: bool = False,
) -> None:
    """Build Sphinx documentation."""
    print("\n" + "=" * 60)
    print("Building Sphinx Documentation")
    print("=" * 60)

    doc_dir = project_root / "doc"
    env = {}

    if skip_cpp:
        print("  C++ docs: skipped")
    else:
        env["READTHEDOCS"] = "True"
        print("  C++ docs: enabled (Doxygen)")

    if r_docs and r_docs.exists():
        env["XGBOOST_R_DOCS"] = str(r_docs)
        print(f"  R docs: {r_docs}")
    elif skip_r:
        env["XGBOOST_SKIP_R_DOCS"] = "1"
        print("  R docs: skipped")

    if jvm_docs and jvm_docs.exists():
        env["XGBOOST_JVM_DOCS"] = str(jvm_docs)
        print(f"  JVM docs: {jvm_docs}")
    elif skip_jvm:
        env["XGBOOST_SKIP_JVM_DOCS"] = "1"
        print("  JVM docs: skipped")

    if not skip_deps:
        run_cmd(["pip", "install", "-r", "requirements.txt"], cwd=doc_dir)

    run_cmd(["make", "html"], cwd=doc_dir, env=env)

    print("\n" + "=" * 60)
    print(f"Output: {doc_dir / '_build/html'}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build XGBoost documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- Subcommand: jvm-lib ---
    p = subparsers.add_parser("jvm-lib", help="Build libxgboost4j.so")
    p.add_argument("--cuda", action="store_true", help="Enable CUDA")
    p.add_argument("--sccache", action="store_true", help="Use sccache")
    p.add_argument("--gpu-arch", help="GPU architecture (e.g., 75)")

    # --- Subcommand: jvm ---
    p = subparsers.add_parser("jvm", help="Build JVM documentation")
    p.add_argument("--branch-name", default="local", help="Branch name for tarball")
    p.add_argument("--no-tarball", action="store_true", help="Skip tarball creation")

    # --- Subcommand: r ---
    p = subparsers.add_parser("r", help="Build R documentation")
    p.add_argument("--branch-name", default="local", help="Branch name for tarball")
    p.add_argument("--no-tarball", action="store_true", help="Skip tarball creation")
    p.add_argument("--r-libs-user", help="R_LIBS_USER path")

    # --- Subcommand: sphinx ---
    p = subparsers.add_parser("sphinx", help="Build Sphinx documentation")
    p.add_argument("--r-docs", help="Path to R docs tarball")
    p.add_argument("--jvm-docs", help="Path to JVM docs tarball")
    p.add_argument("--skip-r", action="store_true", help="Skip R docs")
    p.add_argument("--skip-jvm", action="store_true", help="Skip JVM docs")
    p.add_argument("--skip-cpp", action="store_true", help="Skip C++ docs (Doxygen)")
    p.add_argument("--skip-deps", action="store_true", help="Skip pip install")

    # --- Default (all) ---
    parser.add_argument("--skip-r", action="store_true", help="Skip R docs")
    parser.add_argument("--skip-jvm", action="store_true", help="Skip JVM docs")
    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ docs")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing tarballs")

    args = parser.parse_args()
    project_root = get_project_root()

    # --- Handle subcommands ---
    if args.command == "jvm-lib":
        if not check_dependencies({"jvm-lib"}):
            sys.exit(1)
        if not build_jvm_lib(project_root, args.cuda, args.sccache, args.gpu_arch):
            sys.exit(1)

    elif args.command == "jvm":
        if not check_dependencies({"jvm"}):
            sys.exit(1)
        if not build_jvm_docs(project_root, args.branch_name, not args.no_tarball):
            sys.exit(1)

    elif args.command == "r":
        if not check_dependencies({"r"}):
            sys.exit(1)
        if not build_r_docs(project_root, args.branch_name, not args.no_tarball, args.r_libs_user):
            sys.exit(1)

    elif args.command == "sphinx":
        components = set()
        if not args.skip_cpp:
            components.add("cpp")
        if not check_dependencies(components):
            sys.exit(1)
        r_docs = Path(args.r_docs) if args.r_docs else None
        jvm_docs = Path(args.jvm_docs) if args.jvm_docs else None
        build_sphinx_docs(
            project_root, r_docs, jvm_docs,
            args.skip_r, args.skip_jvm, args.skip_cpp, args.skip_deps
        )

    else:
        # --- Default: build all ---
        components = {"cpp"} if not args.skip_cpp else set()
        if not args.skip_r:
            components.add("r")
        if not args.skip_jvm:
            components.update({"jvm", "jvm-lib"})
        if not check_dependencies(components):
            sys.exit(1)

        r_tarball = None
        jvm_tarball = None

        # Build R docs
        if not args.skip_r:
            existing = project_root / "r-docs-local.tar.bz2"
            if args.reuse and existing.exists():
                print(f"Reusing: {existing}")
                r_tarball = existing
            else:
                r_tarball = build_r_docs(project_root)

        # Build JVM docs
        if not args.skip_jvm:
            existing = project_root / "jvm-docs-local.tar.bz2"
            if args.reuse and existing.exists():
                print(f"Reusing: {existing}")
                jvm_tarball = existing
            else:
                lib_path = project_root / "lib" / "libxgboost4j.so"
                if not lib_path.exists():
                    build_jvm_lib(project_root)
                jvm_tarball = build_jvm_docs(project_root)
                if jvm_tarball:
                    local = project_root / "jvm-docs-local.tar.bz2"
                    shutil.move(jvm_tarball, local)
                    jvm_tarball = local

        # Build Sphinx docs
        build_sphinx_docs(
            project_root, r_tarball, jvm_tarball,
            args.skip_r, args.skip_jvm, args.skip_cpp, args.skip_deps
        )


if __name__ == "__main__":
    main()
