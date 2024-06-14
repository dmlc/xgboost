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
    "USE_HDFS": "OFF",
    "USE_AZURE": "OFF",
    "USE_S3": "OFF",
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
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def run(command, **kwargs):
    print(command)
    subprocess.run(command, shell=True, check=True, env=os.environ, **kwargs)


def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target))
    shutil.copy(source, target)


def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    else:
        return normalized


def native_build(args):
    if sys.platform == "darwin":
        # Enable of your compiler supports OpenMP.
        CONFIG["USE_OPENMP"] = "OFF"
        os.environ["JAVA_HOME"] = (
            subprocess.check_output("/usr/libexec/java_home").strip().decode()
        )

    print("building Java wrapper")
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

        # if enviorment set GPU_ARCH_FLAG
        gpu_arch_flag = os.getenv("GPU_ARCH_FLAG", None)
        if gpu_arch_flag is not None:
            args.append("%s" % gpu_arch_flag)

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
                        print(f"Failed to build with generator: {generator}", e)
                        with cd(os.path.pardir):
                            shutil.rmtree(build_dir)
                            maybe_makedirs(build_dir)
            else:
                run("cmake .. " + " ".join(args))
            run("cmake --build . --config Release" + maybe_parallel_build)

        with cd("demo/CLI/regression"):
            run(f'"{sys.executable}" mapfeat.py')
            run(f'"{sys.executable}" mknfold.py machine.txt 1')

    xgboost4j = "xgboost4j-gpu" if cli_args.use_cuda == "ON" else "xgboost4j"
    xgboost4j_spark = (
        "xgboost4j-spark-gpu" if cli_args.use_cuda == "ON" else "xgboost4j-spark"
    )

    print("copying native library")
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
    output_folder = "{}/src/main/resources/lib/{}/{}".format(
        xgboost4j, os_folder, arch_folder
    )
    maybe_makedirs(output_folder)
    cp("../lib/" + library_name, output_folder)

    print("copying train/test files")
    maybe_makedirs("{}/src/test/resources".format(xgboost4j_spark))
    with cd("../demo/CLI/regression"):
        run(f'"{sys.executable}" mapfeat.py')
        run(f'"{sys.executable}" mknfold.py machine.txt 1')

    for file in glob.glob("../demo/CLI/regression/machine.txt.t*"):
        cp(file, "{}/src/test/resources".format(xgboost4j_spark))
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "{}/src/test/resources".format(xgboost4j_spark))

    maybe_makedirs("{}/src/test/resources".format(xgboost4j))
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "{}/src/test/resources".format(xgboost4j))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-capi-invocation", type=str, choices=["ON", "OFF"], default="OFF"
    )
    parser.add_argument("--use-cuda", type=str, choices=["ON", "OFF"], default="OFF")
    cli_args = parser.parse_args()
    native_build(cli_args)
