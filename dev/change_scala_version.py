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
                f"s/<artifactId>xgboost-jvm_[0-9\\.]*/<artifactId>xgboost-jvm_{args.scala_version}/g",
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
