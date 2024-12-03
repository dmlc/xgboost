"""
Stash an artifact in an S3 bucket for later use

Note. This script takes in all inputs via environment variables
      except the path to the artifact(s).
"""

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit


def resolve(x: Path) -> Path:
    return x.expanduser().resolve()


def path_equals(a: Path, b: Path) -> bool:
    return resolve(a) == resolve(b)


def compute_s3_url(s3_bucket: str, prefix: str, artifact: Path) -> str:
    filename = artifact.name
    relative_path = resolve(artifact).relative_to(Path.cwd())
    if resolve(artifact.parent) == resolve(Path.cwd()):
        full_prefix = prefix
    else:
        full_prefix = f"{prefix}/{str(relative_path.parent)}"
    return f"s3://{s3_bucket}/{full_prefix}/{filename}"


def aws_s3_upload(src: Path, dest: str) -> None:
    cli_args = ["aws", "s3", "cp", "--no-progress", str(src), dest]
    print(" ".join(cli_args))
    subprocess.run(
        cli_args,
        check=True,
        encoding="utf-8",
    )


def aws_s3_download(src: str, dest: Path) -> None:
    cli_args = ["aws", "s3", "cp", "--no-progress", src, str(dest)]
    print(" ".join(cli_args))
    subprocess.run(
        cli_args,
        check=True,
        encoding="utf-8",
    )


def aws_s3_download_with_wildcard(src: str, dest: Path) -> None:
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
    dest_dir = dest.parent
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


def upload(args: argparse.Namespace) -> None:
    print(f"Stashing artifacts to prefix {args.prefix}...")
    for artifact in args.artifacts:
        artifact_path = Path(artifact)
        s3_url = compute_s3_url(args.s3_bucket, args.prefix, artifact_path)
        aws_s3_upload(artifact_path, s3_url)


def download(args: argparse.Namespace) -> None:
    print(f"Unstashing artifacts from prefix {args.prefix}...")
    for artifact in args.artifacts:
        artifact_path = Path(artifact)
        print(f"mkdir -p {str(artifact_path.parent)}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        s3_url = compute_s3_url(args.s3_bucket, args.prefix, artifact_path)
        if "*" in artifact:
            aws_s3_download_with_wildcard(s3_url, artifact_path)
        else:
            aws_s3_download(s3_url, artifact_path)


if __name__ == "__main__":
    # Ensure that the current working directory is the project root
    if not (Path.cwd() / "ops").is_dir() or not path_equals(
        Path(__file__).parent.parent, Path.cwd() / "ops"
    ):
        x = Path(__file__).name
        raise RuntimeError(f"Script {x} must be run at the project's root directory")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        type=str,
        choices=["stash", "unstash"],
        required=True,
        help="Whether to stash or unstash the artifact",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        required=True,
        help="Name of the S3 bucket to store the artifact",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help=(
            "Where the artifact would be stored. The artifact will be stored in "
            "s3://[s3-bucket]/[prefix]."
        ),
    )
    parser.add_argument("artifacts", type=str, nargs="+", metavar="artifact")
    parsed_args = parser.parse_args()
    if parsed_args.command == "stash":
        upload(parsed_args)
    elif parsed_args.command == "unstash":
        download(parsed_args)
