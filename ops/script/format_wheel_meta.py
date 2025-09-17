"""
Script to generate meta.json to store metadata for a nightly build of
XGBoost Python package.
"""

import argparse
import json
import pathlib


def main(args: argparse.Namespace) -> None:
    wheel_path = pathlib.Path(args.wheel_path).expanduser().resolve()
    if not wheel_path.exists():
        raise ValueError(f"Wheel cannot be found at path {wheel_path}")
    if not wheel_path.is_file():
        raise ValueError(f"Path {wheel_path} is not a valid file")
    wheel_name = wheel_path.name

    meta_path = pathlib.Path(args.meta_path)
    if not meta_path.exists():
        raise ValueError(f"Path {meta_path} does not exist")
    if not meta_path.is_dir():
        raise ValueError(f"Path {meta_path} is not a valid directory")

    tokens = wheel_name.split("-")
    assert len(tokens) == 5
    version = tokens[1].split("+")[0]

    meta_info = {
        "wheel_path": f"{args.commit_hash}/{wheel_name}",
        "wheel_name": wheel_name,
        "platform_tag": args.platform_tag,
        "version": version,
        "commit_id": args.commit_hash,
    }
    with open(meta_path / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format meta.json encoding the latest nightly version of the Python wheel"
    )
    parser.add_argument(
        "--wheel-path", type=str, required=True, help="Path to the wheel"
    )
    parser.add_argument(
        "--commit-hash", type=str, required=True, help="Git commit hash"
    )
    parser.add_argument(
        "--platform-tag",
        type=str,
        required=True,
        help="Platform tag (e.g. manylinux_2_28_x86_64)",
    )
    parser.add_argument(
        "--meta-path", type=str, required=True, help="Directory to place meta.json"
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
