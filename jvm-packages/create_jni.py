#!/usr/bin/env python
import argparse
import errno
import glob
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager

# Monkey-patch the API inconsistency between Python2.X and 3.X.
if sys.platform.startswith("linux"):
    sys.platform = "linux"


CONFIG = {
    "USE_OPENMP": "ON",
    "USE_CUDA": "OFF",
    "USE_NCCL": "OFF",
    "JVM_BINDINGS": "ON",
    "LOG_CAPI_INVOCATION": "OFF",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
}


@contextmanager
def cd(path):
    path = normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path, flush=True)
    try:
        yield path
    finally:
        os.chdir(cwd)


def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path, flush=True)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def run(command, **kwargs):
    print(command, flush=True)
    subprocess.run(command, shell=True, check=True, env=os.environ, **kwargs)


def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target), flush=True)
    shutil.copy(source, target)


def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    else:
        return normalized


def native_build(cli_args: argparse.Namespace) -> None:
    CONFIG["USE_OPENMP"] = cli_args.use_openmp
    if sys.platform == "darwin":
        os.environ["JAVA_HOME"] = (
            subprocess.check_output("/usr/libexec/java_home").strip().decode()
        )

    print("building Java wrapper", flush=True)
    with cd(".."):
        build_dir = "build-gpu" if cli_args.use_cuda == "ON" else "build"
        maybe_makedirs(build_dir)

        if sys.platform == "linux":
            maybe_parallel_build = " -- -j $(nproc)"
        elif sys.platform == "win32":
            maybe_parallel_build = ' -- /m /nodeReuse:false "/consoleloggerparameters:ShowCommandLine;Verbosity=minimal"'
        else:
            maybe_parallel_build = ""

        if cli_args.log_capi_invocation == "ON":
            CONFIG["LOG_CAPI_INVOCATION"] = "ON"

        if cli_args.use_cuda == "ON":
            CONFIG["USE_CUDA"] = "ON"
            CONFIG["USE_NCCL"] = "ON"
            CONFIG["USE_DLOPEN_NCCL"] = "OFF"

        args = ["-D{0}:BOOL={1}".format(k, v) for k, v in CONFIG.items()]
        if sys.platform != "win32":
            try:
                subprocess.check_call(["ninja", "--version"])
                args.append("-GNinja")
            except FileNotFoundError:
                pass

        # if enviorment set GPU_ARCH_FLAG
        gpu_arch_flag = os.getenv("GPU_ARCH_FLAG", None)
        if gpu_arch_flag is not None:
            args.append("-DCMAKE_CUDA_ARCHITECTURES=%s" % gpu_arch_flag)

        with cd(build_dir):
            lib_dir = os.path.join(os.pardir, "lib")
            if os.path.exists(lib_dir):
                shutil.rmtree(lib_dir)

            # Same trick as Python build, just test all possible generators.
            if sys.platform == "win32":
                supported_generators = (
                    "",  # empty, decided by cmake
                    '-G"Visual Studio 17 2022" -A x64',
                    '-G"Visual Studio 16 2019" -A x64',
                    '-G"Visual Studio 15 2017" -A x64',
                )
                for generator in supported_generators:
                    try:
                        run("cmake .. " + " ".join(args + [generator]))
                        break
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to build with generator: {generator}", e, flush=True)
                        with cd(os.path.pardir):
                            shutil.rmtree(build_dir)
                            maybe_makedirs(build_dir)
            else:
                run("cmake .. " + " ".join(args))
            run("cmake --build . --config Release" + maybe_parallel_build)


    print("copying native library", flush=True)
    library_name, os_folder = {
        "Windows": ("xgboost4j.dll", "windows"),
        "Darwin": ("libxgboost4j.dylib", "macos"),
        "Linux": ("libxgboost4j.so", "linux"),
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
    output_folder = "xgboost4j/src/main/resources/lib/{}/{}".format(
        os_folder, arch_folder
    )
    maybe_makedirs(output_folder)
    cp("../lib/" + library_name, output_folder)

    print("copying train/test files", flush=True)

    # for xgboost4j
    maybe_makedirs("xgboost4j/src/test/resources")
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "xgboost4j/src/test/resources")

    # for xgboost4j-spark
    maybe_makedirs("xgboost4j-spark/src/test/resources")
    with cd("../demo/CLI/regression"):
        run(f'"{sys.executable}" mapfeat.py')
        run(f'"{sys.executable}" mknfold.py machine.txt 1')
    for file in glob.glob("../demo/CLI/regression/machine.txt.t*"):
        cp(file, "xgboost4j-spark/src/test/resources")
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "xgboost4j-spark/src/test/resources")

    # for xgboost4j-spark-gpu
    if cli_args.use_cuda == "ON":
        maybe_makedirs("xgboost4j-spark-gpu/src/test/resources")
        for file in glob.glob("../demo/data/veterans_lung_cancer.csv"):
            cp(file, "xgboost4j-spark-gpu/src/test/resources")
        cp("xgboost4j-spark/src/test/resources/rank.train.csv", "xgboost4j-spark-gpu/src/test/resources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-capi-invocation", type=str, choices=["ON", "OFF"], default="OFF"
    )
    parser.add_argument("--use-cuda", type=str, choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--use-openmp", type=str, choices=["ON", "OFF"], default="ON")
    cli_args = parser.parse_args()
    native_build(cli_args)
