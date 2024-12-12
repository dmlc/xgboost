"""
Upload an artifact to an S3 bucket for later use
Note. This script takes in all inputs via environment variables
      except the path to the artifact(s).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit


def resolve(x: Path) -> Path:
    return x.expanduser().resolve()


def path_equals(a: Path, b: Path) -> bool:
    return resolve(a) == resolve(b)


def compute_s3_url(*, s3_bucket: str, prefix: str, artifact: str) -> str:
    if prefix == "":
        return f"s3://{s3_bucket}/{artifact}"
    return f"s3://{s3_bucket}/{prefix}/{artifact}"


def aws_s3_upload(*, src: Path, dest: str, make_public: bool) -> None:
    cli_args = ["aws", "s3", "cp", "--no-progress", str(src), dest]
    if make_public:
        cli_args.extend(["--acl", "public-read"])
    print(" ".join(cli_args))
    subprocess.run(
        cli_args,
        check=True,
        encoding="utf-8",
    )


def aws_s3_download(*, src: str, dest_dir: Path) -> None:
    cli_args = ["aws", "s3", "cp", "--no-progress", src, str(dest_dir)]
    print(" ".join(cli_args))
    subprocess.run(
        cli_args,
        check=True,
        encoding="utf-8",
    )


def aws_s3_download_with_wildcard(*, src: str, dest_dir: Path) -> None:
    parsed_src = urlsplit(src)
    src_dir = urlunsplit(
        SplitResult(
            scheme="s3",
            netloc=parsed_src.netloc,
            path=os.path.dirname(parsed_src.path),
            query="",
            fragment="",
        )
    )
    src_glob = os.path.basename(parsed_src.path)
    cli_args = [
        "aws",
        "s3",
        "cp",
        "--recursive",
        "--no-progress",
        "--exclude",
        "'*'",
        "--include",
        src_glob,
        src_dir,
        str(dest_dir),
    ]
    print(" ".join(cli_args))
    subprocess.run(
        cli_args,
        check=True,
        encoding="utf-8",
    )


def upload(*, args: argparse.Namespace) -> None:
    print(f"Uploading artifacts to prefix {args.prefix}...")
    for artifact in args.artifacts:
        artifact_path = Path(artifact)
        s3_url = compute_s3_url(
            s3_bucket=args.s3_bucket, prefix=args.prefix, artifact=artifact_path.name
        )
        aws_s3_upload(src=artifact_path, dest=s3_url, make_public=args.make_public)


def download(*, args: argparse.Namespace) -> None:
    print(f"Downloading artifacts from prefix {args.prefix}...")
    dest_dir = Path(args.dest_dir)
    print(f"mkdir -p {str(dest_dir)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for artifact in args.artifacts:
        s3_url = compute_s3_url(
            s3_bucket=args.s3_bucket, prefix=args.prefix, artifact=artifact
        )
        if "*" in artifact:
            aws_s3_download_with_wildcard(src=s3_url, dest_dir=dest_dir)
        else:
            aws_s3_download(src=s3_url, dest_dir=dest_dir)


if __name__ == "__main__":
    # Ensure that the current working directory is the project root
    if not (Path.cwd() / "ops").is_dir() or not path_equals(
        Path(__file__).parent.parent, Path.cwd() / "ops"
    ):
        x = Path(__file__).name
        raise RuntimeError(f"Script {x} must be run at the project's root directory")

    root_parser = argparse.ArgumentParser()
    subparser_factory = root_parser.add_subparsers(required=True, dest="command")
    parsers = {}
    for command in ["upload", "download"]:
        parsers[command] = subparser_factory.add_parser(command)
        parsers[command].add_argument(
            "--s3-bucket",
            type=str,
            required=True,
            help="Name of the S3 bucket to store the artifact",
        )
        parsers[command].add_argument(
            "--prefix",
            type=str,
            required=True,
            help=(
                "Where the artifact(s) would be stored. The artifact(s) will be stored at "
                "s3://[s3-bucket]/[prefix]/[filename]."
            ),
        )
        parsers[command].add_argument(
            "artifacts",
            type=str,
            nargs="+",
            metavar="artifact",
            help=f"Artifact(s) to {command}",
        )

    parsers["upload"].add_argument(
        "--make-public", action="store_true", help="Make artifact publicly accessible"
    )
    parsers["download"].add_argument(
        "--dest-dir", type=str, required=True, help="Where to download artifact(s)"
    )

    if len(sys.argv) == 1:
        print("1. Upload artifact(s)")
        parsers["upload"].print_help()
        print("\n2. Download artifact(s)")
        parsers["download"].print_help()
        sys.exit(1)

    parsed_args = root_parser.parse_args()
    if parsed_args.command == "upload":
        upload(args=parsed_args)
    elif parsed_args.command == "download":
        download(args=parsed_args)
