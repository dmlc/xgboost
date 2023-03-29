import argparse
import base64
import glob
import hashlib
import os
import pathlib
import re
import shutil
import tempfile

VCOMP140_PATH = "C:\\Windows\\System32\\vcomp140.dll"


def get_sha256sum(path):
    return (
        base64.urlsafe_b64encode(hashlib.sha256(open(path, "rb").read()).digest())
        .decode("latin1")
        .rstrip("=")
    )


def update_record(*, wheel_content_dir, xgboost_version):
    vcomp140_size = os.path.getsize(VCOMP140_PATH)
    vcomp140_hash = get_sha256sum(VCOMP140_PATH)

    record_path = wheel_content_dir / pathlib.Path(
        f"xgboost-{xgboost_version}.dist-info/RECORD"
    )
    with open(record_path, "r") as f:
        record_content = f.read()
    record_content += f"xgboost-{xgboost_version}.data/data/xgboost/vcomp140.dll,"
    record_content += f"sha256={vcomp140_hash},{vcomp140_size}\n"
    with open(record_path, "w") as f:
        f.write(record_content)


def main(args):
    candidates = list(sorted(glob.glob(args.wheel_path)))
    for wheel_path in candidates:
        print(f"Processing wheel {wheel_path}")
        m = re.search(r"xgboost-(.*)\+.*-py3", wheel_path)
        if not m:
            raise ValueError(f"Wheel {wheel_path} has unexpected name")
        version = m.group(1)
        print(f"  Detected version for {wheel_path}: {version}")
        print(f"  Inserting vcomp140.dll into {wheel_path}...")
        with tempfile.TemporaryDirectory() as tempdir:
            wheel_content_dir = pathlib.Path(tempdir) / "wheel_content"
            print(f"    Extract {wheel_path} into {wheel_content_dir}")
            shutil.unpack_archive(
                wheel_path, extract_dir=wheel_content_dir, format="zip"
            )
            data_dir = wheel_content_dir / pathlib.Path(
                f"xgboost-{version}.data/data/xgboost"
            )
            data_dir.mkdir(parents=True, exist_ok=True)

            print(f"    Copy {VCOMP140_PATH} -> {data_dir}")
            shutil.copy(VCOMP140_PATH, data_dir)

            print(f"    Update RECORD")
            update_record(wheel_content_dir=wheel_content_dir, xgboost_version=version)

            print(f"    Content of {wheel_content_dir}:")
            for e in sorted(wheel_content_dir.rglob("*")):
                if e.is_file():
                    r = e.relative_to(wheel_content_dir)
                    print(f"      {r}")

            print(f"    Create new wheel...")
            new_wheel_tmp_path = pathlib.Path(tempdir) / "new_wheel"
            shutil.make_archive(
                str(new_wheel_tmp_path.resolve()),
                format="zip",
                root_dir=wheel_content_dir,
            )
            new_wheel_tmp_path = new_wheel_tmp_path.resolve().with_suffix(".zip")
            new_wheel_tmp_path = new_wheel_tmp_path.rename(
                new_wheel_tmp_path.with_suffix(".whl")
            )
            print(f"    Created new wheel {new_wheel_tmp_path}")

            # Rename the old wheel with suffix .bak
            # The new wheel takes the name of the old wheel
            wheel_path_obj = pathlib.Path(wheel_path).resolve()
            backup_path = wheel_path_obj.with_suffix(".whl.bak")
            print(f"    Rename {wheel_path_obj} -> {backup_path}")
            wheel_path_obj.replace(backup_path)
            print(f"    Rename {new_wheel_tmp_path} -> {wheel_path_obj}")
            new_wheel_tmp_path.replace(wheel_path_obj)

            shutil.rmtree(wheel_content_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wheel_path", type=str, help="Path to wheel (wildcard permitted)"
    )
    args = parser.parse_args()

    main(args)
