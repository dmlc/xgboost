import argparse
import pathlib


def main(args: argparse.Namespace) -> None:
    wheel_path = pathlib.Path(args.wheel_path).expanduser().resolve()
    if not wheel_path.exists():
        raise ValueError(f"Wheel cannot be found at path {wheel_path}")
    if not wheel_path.is_file():
        raise ValueError(f"Path {wheel_path} is not a valid file")
    wheel_dir, wheel_name = wheel_path.parent, wheel_path.name

    tokens = wheel_name.split("-")
    assert len(tokens) == 5
    version = tokens[1].split("+")[0]
    keywords = {
        "pkg_name": tokens[0],
        "version": version,
        "commit_id": args.commit_hash,
        "platform_tag": args.platform_tag,
    }
    new_wheel_name = (
        "{pkg_name}-{version}+{commit_id}-py3-none-{platform_tag}.whl".format(
            **keywords
        )
    )
    new_wheel_path = wheel_dir / new_wheel_name
    print(f"Renaming {wheel_name} to {new_wheel_name}...")
    if new_wheel_name == wheel_name:
        print("Skipping, as the old name is identical to the new name.")
    else:
        if new_wheel_path.is_file():
            new_wheel_path.unlink()
        wheel_path.rename(new_wheel_path)

    filesize = new_wheel_path.stat().st_size / 1024 / 1024  # MiB
    print(f"Wheel size: {filesize:.2f} MiB")

    if filesize > 300:
        raise RuntimeError(
            f"Limit of wheel size set by PyPI is exceeded. {new_wheel_name}: {filesize:.2f} MiB"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format a Python wheel's name using the git commit hash and platform tag"
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
    parsed_args = parser.parse_args()
    main(parsed_args)
