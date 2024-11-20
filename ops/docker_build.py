"""
Wrapper script to build a Docker container with layer caching
"""

import argparse
import itertools
import pathlib
import subprocess
import sys
from typing import Optional

from docker_run import OPS_DIR, fancy_print_cli_args


def parse_build_args(raw_build_args: list[str]) -> dict[str, str]:
    parsed_build_args = dict()
    for arg in raw_build_args:
        try:
            key, value = arg.split("=", maxsplit=1)
        except ValueError as e:
            raise ValueError(
                f"Build argument must be of form KEY=VALUE. Got: {arg}"
            ) from e
        parsed_build_args[key] = value
    return parsed_build_args


def docker_build(
    container_id: str,
    *,
    build_args: dict[str, str],
    dockerfile_path: pathlib.Path,
    docker_context_path: pathlib.Path,
    cache_from: Optional[str],
    cache_to: Optional[str],
) -> None:
    ## Set up command-line arguments to be passed to `docker build`
    # Build args
    docker_build_cli_args = list(
        itertools.chain.from_iterable(
            [["--build-arg", f"{k}={v}"] for k, v in build_args.items()]
        )
    )
    # When building an image using a non-default driver, we need to specify
    # `--load` to load it to the image store.
    # See https://docs.docker.com/build/builders/drivers/
    docker_build_cli_args.append("--load")
    # Layer caching
    if cache_from:
        docker_build_cli_args.extend(["--cache-from", cache_from])
    if cache_to:
        docker_build_cli_args.extend(["--cache-to", cache_to])
    # Remaining CLI args
    docker_build_cli_args.extend(
        [
            "--progress=plain",
            "--ulimit",
            "nofile=1024000:1024000",
            "-t",
            container_id,
            "-f",
            str(dockerfile_path),
            str(docker_context_path),
        ]
    )
    cli_args = ["docker", "build"] + docker_build_cli_args
    fancy_print_cli_args(cli_args)
    subprocess.run(cli_args, check=True, encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    # Dockerfile to be used in docker build
    dockerfile_path = (
        OPS_DIR / "docker" / "dockerfile" / f"Dockerfile.{args.container_def}"
    )
    docker_context_path = OPS_DIR

    build_args = parse_build_args(args.build_arg)

    docker_build(
        args.container_id,
        build_args=build_args,
        dockerfile_path=dockerfile_path,
        docker_context_path=docker_context_path,
        cache_from=args.cache_from,
        cache_to=args.cache_to,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Docker container")
    parser.add_argument(
        "--container-def",
        type=str,
        required=True,
        help=(
            "String uniquely identifying the container definition. The container "
            "definition will be fetched from "
            "docker/dockerfile/Dockerfile.CONTAINER_DEF."
        ),
    )
    parser.add_argument(
        "--container-id",
        type=str,
        required=True,
        help="String ID to assign to the newly built container",
    )
    parser.add_argument(
        "--build-arg",
        type=str,
        default=[],
        action="append",
        help=(
            "Build-time variable(s) to be passed to `docker build`. Each variable "
            "should be specified as a key-value pair in the form KEY=VALUE. "
            "The variables should match the ARG instructions in the Dockerfile. "
            "When passing multiple variables, specify --build-arg multiple times. "
            "Example: --build-arg CUDA_VERSION_ARG=12.5 --build-arg RAPIDS_VERSION_ARG=24.10'"
        ),
    )
    parser.add_argument(
        "--cache-from",
        type=str,
        help="Use an external cache source for the Docker build",
    )
    parser.add_argument(
        "--cache-to",
        type=str,
        help="Export layers from the container to an external cache destination",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    parsed_args = parser.parse_args()
    main(parsed_args)
