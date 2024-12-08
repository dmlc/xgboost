import argparse
import pathlib
import re
import shutil


def main(args: argparse.Namespace) -> None:
    if args.scala_version == "2.12":
        scala_ver = "2.12"
        scala_patchver = "2.12.18"
    elif args.scala_version == "2.13":
        scala_ver = "2.13"
        scala_patchver = "2.13.11"
    else:
        raise ValueError(f"Unsupported Scala version: {args.scala_version}")

    # Clean artifacts
    if args.purge_artifacts:
        for target in pathlib.Path("jvm-packages/").glob("**/target"):
            if target.is_dir():
                print(f"Removing {target}...")
                shutil.rmtree(target)
        for target in pathlib.Path("jvm-packages/").glob("**/*.so"):
            print(f"Removing {target}...")
            target.unlink()

    # Update pom.xml
    for pom in pathlib.Path("jvm-packages/").glob("**/pom.xml"):
        print(f"Updating {pom}...")
        with open(pom, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(pom, "w", encoding="utf-8") as f:
            replaced_scalaver = False
            replaced_scala_binver = False
            for line in lines:
                for artifact in [
                    "xgboost-jvm",
                    "xgboost4j",
                    "xgboost4j-spark",
                    "xgboost4j-spark-gpu",
                    "xgboost4j-flink",
                    "xgboost4j-example",
                ]:
                    line = re.sub(
                        f"<artifactId>{artifact}_[0-9\\.]*",
                        f"<artifactId>{artifact}_{scala_ver}",
                        line,
                    )
                # Only replace the first occurrence of scala.version
                if not replaced_scalaver:
                    line, nsubs = re.subn(
                        r"<scala.version>[0-9\.]*",
                        f"<scala.version>{scala_patchver}",
                        line,
                    )
                    if nsubs > 0:
                        replaced_scalaver = True
                # Only replace the first occurrence of scala.binary.version
                if not replaced_scala_binver:
                    line, nsubs = re.subn(
                        r"<scala.binary.version>[0-9\.]*",
                        f"<scala.binary.version>{scala_ver}",
                        line,
                    )
                    if nsubs > 0:
                        replaced_scala_binver = True
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge-artifacts", action="store_true")
    parser.add_argument(
        "--scala-version",
        type=str,
        required=True,
        help="Version of Scala to use in the JVM packages",
        choices=["2.12", "2.13"],
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
