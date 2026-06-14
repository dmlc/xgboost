#!/usr/bin/env python
"""Build the native XGBoost4J JNI library."""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
JVM_PACKAGES = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "USE_OPENMP": "ON",
    "USE_CUDA": "OFF",
    "USE_NCCL": "OFF",
    "JVM_BINDINGS": "ON",
    "LOG_CAPI_INVOCATION": "OFF",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
}


def run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a shell command."""
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=os.environ)


def mkdir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    print(f"mkdir -p {path}", flush=True)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, target: Path) -> None:
    """Copy a file to a target path or directory."""
    print(f"cp {source} {target}", flush=True)
    shutil.copy(source, target)


def copy_glob(pattern: str, target: Path) -> None:
    """Copy files matching a glob pattern to a target directory."""
    for source in ROOT.glob(pattern):
        copy_file(source, target)


def cmake_config(options: argparse.Namespace) -> dict[str, str]:
    """Create CMake configuration from CLI options."""
    config = DEFAULT_CONFIG.copy()
    config["USE_OPENMP"] = options.use_openmp
    config["USE_NVTX"] = options.use_nvtx
    config["PLUGIN_RMM"] = options.plugin_rmm

    if options.log_capi_invocation == "ON":
        config["LOG_CAPI_INVOCATION"] = "ON"
    if options.use_debug == "ON":
        config["CMAKE_BUILD_TYPE"] = "Debug"
    if options.use_cuda == "ON":
        config["USE_CUDA"] = "ON"
        config["USE_NCCL"] = "ON"
        config["USE_DLOPEN_NCCL"] = "OFF"

    return config


def cmake_args(config: dict[str, str]) -> list[str]:
    """Create CMake command line arguments."""
    args = [f"-D{k}:BOOL={v}" for k, v in config.items()]

    if sys.platform != "win32":
        try:
            subprocess.check_call(["ninja", "--version"])
            args.append("-GNinja")
        except FileNotFoundError:
            pass

    # Set GPU_ARCH_FLAG to override the CUDA architectures.
    gpu_arch_flag = os.getenv("GPU_ARCH_FLAG")
    if gpu_arch_flag:
        args.append(f"-DCMAKE_CUDA_ARCHITECTURES={gpu_arch_flag}")

    return args


def windows_generators() -> tuple[list[str], ...]:
    """Return CMake generator arguments to try on Windows."""
    return (
        [],  # Let CMake decide.
        ["-G", "Visual Studio 18 2026", "-A", "x64"],
        ["-G", "Visual Studio 17 2022", "-A", "x64"],
        ["-G", "Visual Studio 16 2019", "-A", "x64"],
        ["-G", "Visual Studio 15 2017", "-A", "x64"],
    )


def configure(config_args: list[str], build_dir: Path) -> None:
    """Configure the CMake build."""
    if sys.platform == "win32":
        for generator in windows_generators():
            try:
                run(["cmake", str(ROOT), *config_args, *generator], cwd=build_dir)
                return
            except subprocess.CalledProcessError as err:
                print(
                    f"Failed to build with generator: {shlex.join(generator)}",
                    err,
                    flush=True,
                )
                shutil.rmtree(build_dir)
                mkdir(build_dir)
        raise RuntimeError("None of the supported CMake generators worked.")

    run(["cmake", str(ROOT), *config_args], cwd=build_dir)


def build(config: dict[str, str], build_dir: Path) -> None:
    """Build the native library."""
    lib_dir = ROOT / "lib"
    if lib_dir.exists():
        shutil.rmtree(lib_dir)

    configure(cmake_args(config), build_dir)

    build_args = ["cmake", "--build", ".", "--config", "Release"]
    if sys.platform == "linux":
        build_args.extend(["--", "-j", str(os.cpu_count() or 1)])
    elif sys.platform == "win32":
        build_args.extend(
            [
                "--",
                "/m",
                "/nodeReuse:false",
                "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal",
            ]
        )
    run(build_args, cwd=build_dir)


def copy_native_library() -> None:
    """Copy the native library into the JVM package resources."""
    library_name, os_folder = {
        "Windows": ("xgboost4j.dll", "windows"),
        "Darwin": ("libxgboost4j.dylib", "macos"),
        "Linux": ("libxgboost4j.so", "linux"),
        "FreeBSD": ("libxgboost4j.so", "freebsd"),
        "SunOS": ("libxgboost4j.so", "solaris"),
    }[platform.system()]
    arch_folder = {
        "x86_64": "x86_64",  # on Linux & macOS x86_64
        "amd64": "x86_64",  # on Windows x86_64
        "i86pc": "x86_64",  # on Solaris x86_64
        "sun4v": "sparc",  # on Solaris sparc
        "arm64": "aarch64",  # on macOS & Windows ARM 64-bit
        "aarch64": "aarch64",
    }[platform.machine().lower()]

    output_folder = (
        JVM_PACKAGES / "xgboost4j/src/main/resources/lib" / os_folder / arch_folder
    )
    mkdir(output_folder)
    copy_file(ROOT / "lib" / library_name, output_folder)


def copy_test_resources(*, use_cuda: bool) -> None:
    """Copy training data used by JVM package tests."""
    xgboost4j_resources = JVM_PACKAGES / "xgboost4j/src/test/resources"
    mkdir(xgboost4j_resources)
    copy_glob("demo/data/agaricus.*", xgboost4j_resources)

    xgboost4j_spark_resources = JVM_PACKAGES / "xgboost4j-spark/src/test/resources"
    mkdir(xgboost4j_spark_resources)

    regression_dir = ROOT / "demo/data/regression"
    run([sys.executable, "mapfeat.py"], cwd=regression_dir)
    run([sys.executable, "mknfold.py", "machine.txt", "1"], cwd=regression_dir)

    copy_glob("demo/data/regression/machine.txt.t*", xgboost4j_spark_resources)
    copy_glob("demo/data/agaricus.*", xgboost4j_spark_resources)

    if use_cuda:
        xgboost4j_spark_gpu_resources = (
            JVM_PACKAGES / "xgboost4j-spark-gpu/src/test/resources"
        )
        mkdir(xgboost4j_spark_gpu_resources)
        copy_glob("demo/data/veterans_lung_cancer.csv", xgboost4j_spark_gpu_resources)
        copy_file(
            xgboost4j_spark_resources / "rank.train.csv",
            xgboost4j_spark_gpu_resources,
        )


def native_build(options: argparse.Namespace) -> None:
    """Build and copy the native JNI library and its test resources."""
    if sys.platform == "darwin":
        os.environ["JAVA_HOME"] = (
            subprocess.check_output(["/usr/libexec/java_home"]).strip().decode()
        )

    print("building Java wrapper", flush=True)
    build_dir = ROOT / ("build-gpu" if options.use_cuda == "ON" else "build")
    mkdir(build_dir)
    build(cmake_config(options), build_dir)

    print("copying native library", flush=True)
    copy_native_library()

    print("copying train/test files", flush=True)
    copy_test_resources(use_cuda=options.use_cuda == "ON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-capi-invocation", type=str, choices=["ON", "OFF"], default="OFF"
    )
    parser.add_argument("--use-cuda", type=str, choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--use-openmp", type=str, choices=["ON", "OFF"], default="ON")
    parser.add_argument("--use-debug", type=str, choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--use-nvtx", type=str, choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--plugin-rmm", type=str, choices=["ON", "OFF"], default="OFF")
    parsed_args = parser.parse_args()
    native_build(parsed_args)
