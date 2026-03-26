#!/usr/bin/env python3
"""Query the latest version tag from a GitHub repository.

Usage:
    python3 ops/pipeline/query-latest-version.py --repo NVIDIA/cccl
    python3 ops/pipeline/query-latest-version.py --repo rapidsai/rmm --extract-minor
"""

import argparse
import re
import subprocess
import sys

from packaging.version import InvalidVersion, Version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the latest version tag from a GitHub repository."
    )
    parser.add_argument(
        "--repo", required=True, help="GitHub repo, e.g. NVIDIA/cccl or rapidsai/rmm"
    )
    parser.add_argument(
        "--extract-minor",
        action="store_true",
        help="Extract only the major.minor version (e.g. 25.06 from v25.06.00a1)",
    )
    args = parser.parse_args()

    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{args.repo}/tags",
            "--paginate",
            "--jq",
            ".[].name",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    versions = []
    for tag in result.stdout.strip().split("\n"):
        tag = tag.strip()
        if not tag:
            continue
        try:
            versions.append((tag, Version(tag)))
        except InvalidVersion:
            pass

    if not versions:
        print(f"No valid version tags found in {args.repo}", file=sys.stderr)
        sys.exit(1)

    latest_tag = max(versions, key=lambda x: x[1])[0]

    if args.extract_minor:
        m = re.search(r"v?(\d+\.\d+)", latest_tag)
        if m:
            print(m.group(1))
        else:
            print(latest_tag)
    else:
        print(latest_tag)


if __name__ == "__main__":
    main()
