import argparse
import pathlib
import re
import shutil

try:
    import sh
except ImportError as e:
    raise ImportError(
        "Please install sh in your Python environment.\n"
        " - Pip: pip install sh\n"
        " - Conda: conda install -c conda-forge sh"
    ) from e


def main(args):
    if args.scala_version == "2.12":
        scala_ver = "2.12"
        scala_patchver = "2.12.18"
    elif args.scala_version == "2.13":
        scala_ver = "2.13"
        scala_patchver = "2.13.11"
    else:
        raise ValueError(f"Unsupported Scala version: {args.scala_version}")

    # Clean artifacts
    for target in pathlib.Path("jvm-packages/").glob("**/target"):
        if target.is_dir():
            print(f"Removing {target}...")
            shutil.rmtree(target)

    # Update pom.xml
    for pom in pathlib.Path("jvm-packages/").glob("**/pom.xml"):
        print(f"Updating {pom}...")
        sh.sed(
            [
                "-i",
                "-e",
                f"s/<artifactId>xgboost-jvm_[0-9\\.]*/<artifactId>xgboost-jvm_{scala_ver}/g",
                "-e",
                # Only replace the first occurrence of scala.version
                f"0,/<scala.version>/ s/<scala.version>[0-9\\.]*/<scala.version>{scala_patchver}/",
                "-e",
                # Only replace the first occurrence of scala.binary.version
                f"0,/<scala.binary.version>/ s/<scala.binary.version>[0-9\\.]*/<scala.binary.version>{scala_ver}/",
                str(pom),
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scala-version",
        type=str,
        required=True,
        help="Version of Scala to use in the JVM packages",
        choices=["2.12", "2.13"],
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
